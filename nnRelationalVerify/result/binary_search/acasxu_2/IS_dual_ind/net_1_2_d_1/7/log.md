## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 853.294441270257


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450)
1: (-228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977)
2: (-233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266)
3: (-369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684)
4: (-372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588)

## BASE Result
execution time: IAR + LP analysis = 1.93 + 2.02 = 3.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031189


# Binary Search by BASE starts (time budget: 1196.04 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=991.4861450195312
rel_dist={0: [-853.3029989672884, 853.3029989672882]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=991.4861450195312
rel_dist={0: [-853.3028264006657, 853.3028264006657]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=991.4861450195312
rel_dist={0: [-853.3026694719146, 853.3026694719144]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=991.4861450195312
rel_dist={0: [-853.302527144071, 853.3025271440708]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=991.4861450195312
rel_dist={0: [-853.30240514817, 853.3024051481698]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=991.4861450195312
rel_dist={0: [-853.3023271019428, 853.3023271019429]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=991.4861450195312
rel_dist={0: [-853.302286494937, 853.3022864949367]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=991.4861450195312
rel_dist={0: [-853.3022647116577, 853.3022647116577]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=991.4861450195312
rel_dist={0: [-853.3022528297406, 853.3022528297406]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=991.4861450195312
rel_dist={0: [-853.3022468887846, 853.3022468887846]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=991.4861450195312
rel_dist={0: [-853.3022439175052, 853.3022439183108]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=991.4861450195312
rel_dist={0: [-853.3022424330826, 853.3022424330825]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=991.4861450195312
rel_dist={0: [-853.302241690486, 853.302241690486]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=991.4861450195312
rel_dist={0: [-853.3022413192223, 853.302241319222]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=991.4861450195312
rel_dist={0: [-853.3022411337168, 853.3022411324764]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=991.4861450195312
rel_dist={0: [-853.3022410436956, 853.302241041074]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=991.4861450195312
rel_dist={0: [-853.3022409975672, 853.3022409978428]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=991.4861450195312
rel_dist={0: [-853.3022409760595, 853.3022409828156]}

## Binary Search Result
Binary search time: 91.58 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1104.46 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025811, upper bound: 853.3028765
time: 0.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -853.3025811, upper bound: 853.3028765
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -182.7681427, 808.7180176, -987.9567871, 976.7568970
1: -224.1477814, 903.8054199, -228.5348969, 920.6278076, -1144.7756348, 1132.3403320
2: -228.6609802, 895.1587524, -233.1255188, 911.7817383, -1140.4427490, 1128.2843018
3: -362.4073792, 939.4089966, -369.3415222, 956.9161377, -1319.3234863, 1308.7504883
4: -365.4209595, 895.6079712, -372.4694519, 912.2335815, -1277.6544189, 1268.0772705

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -183.7534332, 814.8997803, -182.7681427, 808.7180176, -992.4713745, 997.6679077
1: -229.7754364, 927.4383545, -228.5348969, 920.6278076, -1150.4031982, 1155.9730225
2: -234.2279053, 918.4588013, -233.1255188, 911.7817383, -1146.0096436, 1151.5842285
3: -371.8710327, 963.9637451, -369.3415222, 956.9161377, -1328.7871094, 1333.3052979
4: -374.7720337, 918.7300415, -372.4694519, 912.2335815, -1287.0056152, 1291.1993408

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -179.2388916, 793.9887695, -973.2276001, 973.2276001
1: -224.1477814, 903.8054199, -224.1477814, 903.8054199, -1127.9532471, 1127.9532471
2: -228.6609802, 895.1587524, -228.6609802, 895.1587524, -1123.8197021, 1123.8197021
3: -362.4073792, 939.4089966, -362.4073792, 939.4089966, -1301.8164062, 1301.8164062
4: -365.4209595, 895.6079712, -365.4209595, 895.6079712, -1261.0289307, 1261.0289307

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -183.7534332, 814.8997803, -994.1386108, 977.7421265
1: -224.1477814, 903.8054199, -229.7754364, 927.4383545, -1151.5861816, 1133.5808105
2: -228.6609802, 895.1587524, -234.2279053, 918.4588013, -1147.1196289, 1129.3867188
3: -362.4073792, 939.4089966, -371.8710327, 963.9637451, -1326.3710938, 1311.2800293
4: -365.4209595, 895.6079712, -374.7720337, 918.7300415, -1284.1510010, 1270.3800049

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 1.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -183.7534332, 814.8997803, -179.2388916, 793.9887695, -977.7421265, 994.1386108
1: -229.7754364, 927.4383545, -224.1477814, 903.8054199, -1133.5808105, 1151.5861816
2: -234.2279053, 918.4588013, -228.6609802, 895.1587524, -1129.3867188, 1147.1197510
3: -371.8710327, 963.9637451, -362.4073792, 939.4089966, -1311.2800293, 1326.3710938
4: -374.7720337, 918.7300415, -365.4209595, 895.6079712, -1270.3800049, 1284.1510010

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021835, upper bound: 853.3020092
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 1.04 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -183.7534332, 814.8997803, -183.7534332, 814.8997803, -998.6531982, 998.6531982
1: -229.7754364, 927.4383545, -229.7754364, 927.4383545, -1157.2137451, 1157.2137451
2: -234.2279053, 918.4588013, -234.2279053, 918.4588013, -1152.6865234, 1152.6865234
3: -371.8710327, 963.9637451, -371.8710327, 963.9637451, -1335.8347168, 1335.8347168
4: -374.7720337, 918.7300415, -374.7720337, 918.7300415, -1293.5020752, 1293.5020752

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021835, upper bound: 853.3020092
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3021835, upper bound: 853.3020092
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3021835, upper bound: 853.3020092
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.77
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -179.2388916, 793.9887695, -951.9316406, 882.5560303
1: -197.7702942, 800.6807251, -224.1477814, 903.8054199, -1101.5756836, 1024.8284912
2: -202.2394409, 793.0823975, -228.6609802, 895.1587524, -1097.3981934, 1021.7433472
3: -319.9077759, 831.7758789, -362.4073792, 939.4089966, -1259.3167725, 1194.1831055
4: -323.1899719, 793.5729980, -365.4209595, 895.6079712, -1218.7979736, 1158.9938965

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -179.2388916, 793.9887695, -965.4863281, 939.6948242
1: -214.4616547, 865.6424561, -224.1477814, 903.8054199, -1118.2670898, 1089.7902832
2: -218.7769775, 857.2067871, -228.6609802, 895.1587524, -1113.9357910, 1085.8676758
3: -346.8608398, 899.7705688, -362.4073792, 939.4089966, -1286.2697754, 1262.1779785
4: -349.8637695, 857.6834106, -365.4209595, 895.6079712, -1245.4715576, 1223.1043701

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -183.7534332, 814.8997803, -972.8427124, 887.0705566
1: -197.7702942, 800.6807251, -229.7754364, 927.4383545, -1125.2083740, 1030.4561768
2: -202.2394409, 793.0823975, -234.2279053, 918.4588013, -1120.6982422, 1027.3103027
3: -319.9077759, 831.7758789, -371.8710327, 963.9637451, -1283.8715820, 1203.6467285
4: -323.1899719, 793.5729980, -374.7720337, 918.7300415, -1241.9200439, 1168.3449707

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -183.7534332, 814.8997803, -986.3973999, 944.2092896
1: -214.4616547, 865.6424561, -229.7754364, 927.4383545, -1141.8997803, 1095.4178467
2: -218.7769775, 857.2067871, -234.2279053, 918.4588013, -1137.2357178, 1091.4346924
3: -346.8608398, 899.7705688, -371.8710327, 963.9637451, -1310.8245850, 1271.6416016
4: -349.8637695, 857.6834106, -374.7720337, 918.7300415, -1268.5936279, 1232.4554443

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -179.2388916, 793.9887695, -959.3354492, 916.1560669
1: -207.0206451, 838.7129517, -224.1477814, 903.8054199, -1110.8260498, 1062.8607178
2: -211.4122925, 830.5660400, -228.6609802, 895.1587524, -1106.5710449, 1059.2268066
3: -335.2691040, 871.3525391, -362.4073792, 939.4089966, -1274.6781006, 1233.7598877
4: -338.4065247, 830.8461914, -365.4209595, 895.6079712, -1234.0144043, 1196.2670898

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -179.2388916, 793.9887695, -969.7165527, 959.4926758
1: -219.7311096, 888.0158081, -224.1477814, 903.8054199, -1123.5364990, 1112.1635742
2: -223.9974518, 879.2314453, -228.6609802, 895.1587524, -1119.1562500, 1107.8923340
3: -355.7673950, 922.9997559, -362.4073792, 939.4089966, -1295.1763916, 1285.4071045
4: -358.6697693, 879.5055542, -365.4209595, 895.6079712, -1254.2775879, 1244.9265137

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -183.7534332, 814.8997803, -980.2465210, 920.6706543
1: -207.0206451, 838.7129517, -229.7754364, 927.4383545, -1134.4589844, 1068.4884033
2: -211.4122925, 830.5660400, -234.2279053, 918.4588013, -1129.8710938, 1064.7938232
3: -335.2691040, 871.3525391, -371.8710327, 963.9637451, -1299.2329102, 1243.2236328
4: -338.4065247, 830.8461914, -374.7720337, 918.7300415, -1257.1365967, 1205.6181641

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -183.7534332, 814.8997803, -990.6276245, 964.0072632
1: -219.7311096, 888.0158081, -229.7754364, 927.4383545, -1147.1693115, 1117.7912598
2: -223.9974518, 879.2314453, -234.2279053, 918.4588013, -1142.4562988, 1113.4592285
3: -355.7673950, 922.9997559, -371.8710327, 963.9637451, -1319.7312012, 1294.8708496
4: -358.6697693, 879.5055542, -374.7720337, 918.7300415, -1277.3996582, 1254.2775879

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.62 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3030943, upper bound: 853.3030943
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3025225, upper bound: 853.3024013
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -157.9430389, 703.3171387, -861.2600708, 861.2600708
1: -197.7702942, 800.6807251, -197.7702942, 800.6807251, -998.4510498, 998.4510498
2: -202.2394409, 793.0823975, -202.2394409, 793.0823975, -995.3218384, 995.3218384
3: -319.9077759, 831.7758789, -319.9077759, 831.7758789, -1151.6832275, 1151.6833496
4: -323.1899719, 793.5729980, -323.1899719, 793.5729980, -1116.7629395, 1116.7629395

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014826, upper bound: 853.3003272
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2996511
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -171.4976959, 760.4559326, -918.3988647, 874.8148193
1: -197.7702942, 800.6807251, -214.4616547, 865.6424561, -1063.4125977, 1015.1423340
2: -202.2394409, 793.0823975, -218.7769775, 857.2067871, -1059.4461670, 1011.8593750
3: -319.9077759, 831.7758789, -346.8608398, 899.7705688, -1219.6783447, 1178.6365967
4: -323.1899719, 793.5729980, -349.8637695, 857.6834106, -1180.8734131, 1143.4365234

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014826, upper bound: 853.3003272
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2996511
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -157.9430389, 703.3171387, -874.8147583, 918.3988647
1: -214.4616547, 865.6424561, -197.7702942, 800.6807251, -1015.1423340, 1063.4125977
2: -218.7769775, 857.2067871, -202.2394409, 793.0823975, -1011.8593750, 1059.4461670
3: -346.8608398, 899.7705688, -319.9077759, 831.7758789, -1178.6365967, 1219.6783447
4: -349.8637695, 857.6834106, -323.1899719, 793.5729980, -1143.4365234, 1180.8734131

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023827, upper bound: 853.3016342
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -171.4976959, 760.4559326, -931.9534912, 931.9534912
1: -214.4616547, 865.6424561, -214.4616547, 865.6424561, -1080.1040039, 1080.1041260
2: -218.7769775, 857.2067871, -218.7769775, 857.2067871, -1075.9836426, 1075.9836426
3: -346.8608398, 899.7705688, -346.8608398, 899.7705688, -1246.6313477, 1246.6313477
4: -349.8637695, 857.6834106, -349.8637695, 857.6834106, -1207.5471191, 1207.5471191

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023827, upper bound: 853.3016342
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -165.3467407, 736.9172363, -894.8601685, 868.6638794
1: -197.7702942, 800.6807251, -207.0206451, 838.7129517, -1036.4832764, 1007.7013550
2: -202.2394409, 793.0823975, -211.4122925, 830.5660400, -1032.8054199, 1004.4946899
3: -319.9077759, 831.7758789, -335.2691040, 871.3525391, -1191.2602539, 1167.0446777
4: -323.1899719, 793.5729980, -338.4065247, 830.8461914, -1154.0361328, 1131.9794922

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985199, upper bound: 853.2966904
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974704, upper bound: 853.2960143
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -175.7278595, 780.2539062, -938.1967773, 879.0449829
1: -197.7702942, 800.6807251, -219.7311096, 888.0158081, -1085.7858887, 1020.4118042
2: -202.2394409, 793.0823975, -223.9974518, 879.2314453, -1081.4708252, 1017.0798340
3: -319.9077759, 831.7758789, -355.7673950, 922.9997559, -1242.9074707, 1187.5429688
4: -323.1899719, 793.5729980, -358.6697693, 879.5055542, -1202.6955566, 1152.2426758

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985199, upper bound: 853.2966904
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974704, upper bound: 853.2960143
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -165.3467407, 736.9172363, -908.4148560, 925.8026733
1: -214.4616547, 865.6424561, -207.0206451, 838.7129517, -1053.1745605, 1072.6630859
2: -218.7769775, 857.2067871, -211.4122925, 830.5660400, -1049.3428955, 1068.6191406
3: -346.8608398, 899.7705688, -335.2691040, 871.3525391, -1218.2133789, 1235.0395508
4: -349.8637695, 857.6834106, -338.4065247, 830.8461914, -1180.7098389, 1196.0899658

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2880844, upper bound: 853.2684921
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2872354, upper bound: 853.2679908
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -175.7278595, 780.2539062, -951.7514648, 936.1837158
1: -214.4616547, 865.6424561, -219.7311096, 888.0158081, -1102.4772949, 1085.3735352
2: -218.7769775, 857.2067871, -223.9974518, 879.2314453, -1098.0083008, 1081.2042236
3: -346.8608398, 899.7705688, -355.7673950, 922.9997559, -1269.8605957, 1255.5378418
4: -349.8637695, 857.6834106, -358.6697693, 879.5055542, -1229.3692627, 1216.3531494

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2880844, upper bound: 853.2684921
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2872354, upper bound: 853.2679908
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -157.9430389, 703.3171387, -868.6638794, 894.8601685
1: -207.0206451, 838.7129517, -197.7702942, 800.6807251, -1007.7013550, 1036.4832764
2: -211.4122925, 830.5660400, -202.2394409, 793.0823975, -1004.4946899, 1032.8054199
3: -335.2691040, 871.3525391, -319.9077759, 831.7758789, -1167.0446777, 1191.2602539
4: -338.4065247, 830.8461914, -323.1899719, 793.5729980, -1131.9794922, 1154.0361328

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012647, upper bound: 853.2998140
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987433, upper bound: 853.2980902
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -171.4976959, 760.4559326, -925.8026123, 908.4148560
1: -207.0206451, 838.7129517, -214.4616547, 865.6424561, -1072.6630859, 1053.1745605
2: -211.4122925, 830.5660400, -218.7769775, 857.2067871, -1068.6191406, 1049.3428955
3: -335.2691040, 871.3525391, -346.8608398, 899.7705688, -1235.0395508, 1218.2133789
4: -338.4065247, 830.8461914, -349.8637695, 857.6834106, -1196.0899658, 1180.7098389

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012647, upper bound: 853.2998140
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987433, upper bound: 853.2980902
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -157.9430389, 703.3171387, -879.0449829, 938.1967773
1: -219.7311096, 888.0158081, -197.7702942, 800.6807251, -1020.4118042, 1085.7858887
2: -223.9974518, 879.2314453, -202.2394409, 793.0823975, -1017.0798340, 1081.4708252
3: -355.7673950, 922.9997559, -319.9077759, 831.7758789, -1187.5429688, 1242.9074707
4: -358.6697693, 879.5055542, -323.1899719, 793.5729980, -1152.2426758, 1202.6955566

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018109, upper bound: 853.3010664
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -171.4976959, 760.4559326, -936.1837769, 951.7514648
1: -219.7311096, 888.0158081, -214.4616547, 865.6424561, -1085.3735352, 1102.4772949
2: -223.9974518, 879.2314453, -218.7769775, 857.2067871, -1081.2042236, 1098.0083008
3: -355.7673950, 922.9997559, -346.8608398, 899.7705688, -1255.5378418, 1269.8605957
4: -358.6697693, 879.5055542, -349.8637695, 857.6834106, -1216.3531494, 1229.3692627

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018109, upper bound: 853.3010664
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -165.3467407, 736.9172363, -902.2639771, 902.2639771
1: -207.0206451, 838.7129517, -207.0206451, 838.7129517, -1045.7336426, 1045.7336426
2: -211.4122925, 830.5660400, -211.4122925, 830.5660400, -1041.9782715, 1041.9782715
3: -335.2691040, 871.3525391, -335.2691040, 871.3525391, -1206.6215820, 1206.6215820
4: -338.4065247, 830.8461914, -338.4065247, 830.8461914, -1169.2526855, 1169.2526855

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983020, upper bound: 853.2961772
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961084, upper bound: 853.2947559
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -175.7278595, 780.2539062, -945.6005859, 912.6450806
1: -207.0206451, 838.7129517, -219.7311096, 888.0158081, -1095.0364990, 1058.4440918
2: -211.4122925, 830.5660400, -223.9974518, 879.2314453, -1090.6437988, 1054.5634766
3: -335.2691040, 871.3525391, -355.7673950, 922.9997559, -1258.2687988, 1227.1198730
4: -338.4065247, 830.8461914, -358.6697693, 879.5055542, -1217.9121094, 1189.5159912

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983020, upper bound: 853.2961772
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961084, upper bound: 853.2947559
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -165.3467407, 736.9172363, -912.6450806, 945.6005859
1: -219.7311096, 888.0158081, -207.0206451, 838.7129517, -1058.4440918, 1095.0364990
2: -223.9974518, 879.2314453, -211.4122925, 830.5660400, -1054.5634766, 1090.6437988
3: -355.7673950, 922.9997559, -335.2691040, 871.3525391, -1227.1198730, 1258.2687988
4: -358.6697693, 879.5055542, -338.4065247, 830.8461914, -1189.5158691, 1217.9121094

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2880864, upper bound: 853.2680664
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -175.7278595, 780.2539062, -955.9816895, 955.9816895
1: -219.7311096, 888.0158081, -219.7311096, 888.0158081, -1107.7467041, 1107.7467041
2: -223.9974518, 879.2314453, -223.9974518, 879.2314453, -1103.2288818, 1103.2288818
3: -355.7673950, 922.9997559, -355.7673950, 922.9997559, -1278.7670898, 1278.7670898
4: -358.6697693, 879.5055542, -358.6697693, 879.5055542, -1238.1752930, 1238.1752930

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2880864, upper bound: 853.2680664
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.08 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3014826, upper bound: 853.3003272
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2996511
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3014826, upper bound: 853.3003272
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2996511
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3023827, upper bound: 853.3016342
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3023827, upper bound: 853.3016342
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2985199, upper bound: 853.2966904
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2974704, upper bound: 853.2960143
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2985199, upper bound: 853.2966904
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2974704, upper bound: 853.2960143
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2880844, upper bound: 853.2684921
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2872354, upper bound: 853.2679908
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2880844, upper bound: 853.2684921
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2872354, upper bound: 853.2679908
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3012647, upper bound: 853.2998140
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2987433, upper bound: 853.2980902
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3012647, upper bound: 853.2998140
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2987433, upper bound: 853.2980902
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3018109, upper bound: 853.3010664
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.3018109, upper bound: 853.3010664
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2983020, upper bound: 853.2961772
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2961084, upper bound: 853.2947559
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2983020, upper bound: 853.2961772
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2961084, upper bound: 853.2947559
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2880864, upper bound: 853.2680664
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2880864, upper bound: 853.2680664
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -157.9430389, 703.3171387, -841.2138672, 774.2240601
1: -172.6246338, 701.4801025, -197.7702942, 800.6807251, -973.3053589, 899.2503662
2: -176.5765839, 694.5836182, -202.2394409, 793.0823975, -969.6589966, 896.8230591
3: -279.6128540, 728.3481445, -319.9077759, 831.7758789, -1111.3886719, 1048.2556152
4: -282.3982544, 694.6760254, -323.1899719, 793.5729980, -1075.9710693, 1017.8659668

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -157.9430389, 703.3171387, -858.2327271, 847.9165649
1: -193.9920044, 785.5104980, -197.7702942, 800.6807251, -994.6727295, 983.2807617
2: -198.3775940, 778.0156860, -202.2394409, 793.0823975, -991.4599609, 980.2551270
3: -313.7880249, 816.0270386, -319.9077759, 831.7758789, -1145.5639648, 1135.9344482
4: -317.0524902, 778.5471191, -323.1899719, 793.5729980, -1110.6252441, 1101.7370605

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -171.4976959, 760.4559326, -898.3525391, 787.7787476
1: -172.6246338, 701.4801025, -214.4616547, 865.6424561, -1038.2669678, 915.9417114
2: -176.5765839, 694.5836182, -218.7769775, 857.2067871, -1033.7833252, 913.3605957
3: -279.6128540, 728.3481445, -346.8608398, 899.7705688, -1179.3834229, 1075.2088623
4: -282.3982544, 694.6760254, -349.8637695, 857.6834106, -1140.0815430, 1044.5397949

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998366
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001635
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -171.4976959, 760.4559326, -915.3715210, 861.4713135
1: -193.9920044, 785.5104980, -214.4616547, 865.6424561, -1059.6342773, 999.9721680
2: -198.3775940, 778.0156860, -218.7769775, 857.2067871, -1055.5842285, 996.7926636
3: -313.7880249, 816.0270386, -346.8608398, 899.7705688, -1213.5585938, 1162.8878174
4: -317.0524902, 778.5471191, -349.8637695, 857.6834106, -1174.7358398, 1128.4106445

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002049, upper bound: 853.2993242
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2996511
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -157.9430389, 703.3171387, -870.0597534, 898.1791382
1: -208.5561523, 842.6534424, -197.7702942, 800.6807251, -1009.2368164, 1040.4235840
2: -212.7993317, 834.3977051, -202.2394409, 793.0823975, -1005.8817139, 1036.6372070
3: -337.4005737, 875.8064575, -319.9077759, 831.7758789, -1169.1763916, 1195.7141113
4: -340.4402466, 834.8568115, -323.1899719, 793.5729980, -1134.0131836, 1158.0467529

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998366, upper bound: 853.3011653
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -157.9430389, 703.3171387, -877.5350342, 932.1248779
1: -217.9788666, 881.2302246, -197.7702942, 800.6807251, -1018.6594849, 1079.0003662
2: -222.4038544, 872.6578369, -202.2394409, 793.0823975, -1015.4862671, 1074.8972168
3: -352.7824707, 915.8316650, -319.9077759, 831.7758789, -1184.5583496, 1235.7390137
4: -355.6772156, 873.3172607, -323.1899719, 793.5729980, -1149.2502441, 1196.5072021

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001635, upper bound: 853.3013934
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -171.4976959, 760.4559326, -927.1985474, 911.7338257
1: -208.5561523, 842.6534424, -214.4616547, 865.6424561, -1074.1986084, 1057.1151123
2: -212.7993317, 834.3977051, -218.7769775, 857.2067871, -1070.0061035, 1053.1744385
3: -337.4005737, 875.8064575, -346.8608398, 899.7705688, -1237.1711426, 1222.6672363
4: -340.4402466, 834.8568115, -349.8637695, 857.6834106, -1198.1236572, 1184.7203369

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -171.4976959, 760.4559326, -934.6737671, 945.6795044
1: -217.9788666, 881.2302246, -214.4616547, 865.6424561, -1083.6212158, 1095.6918945
2: -222.4038544, 872.6578369, -218.7769775, 857.2067871, -1079.6105957, 1091.4346924
3: -352.7824707, 915.8316650, -346.8608398, 899.7705688, -1252.5529785, 1262.6922607
4: -355.6772156, 873.3172607, -349.8637695, 857.6834106, -1213.3605957, 1223.1809082

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -165.3467407, 736.9172363, -874.8139038, 781.6278687
1: -172.6246338, 701.4801025, -207.0206451, 838.7129517, -1011.3375854, 908.5006714
2: -176.5765839, 694.5836182, -211.4122925, 830.5660400, -1007.1425171, 905.9959106
3: -279.6128540, 728.3481445, -335.2691040, 871.3525391, -1150.9653320, 1063.6169434
4: -282.3982544, 694.6760254, -338.4065247, 830.8461914, -1113.2443848, 1033.0825195

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -165.3467407, 736.9172363, -891.8328247, 855.3203735
1: -193.9920044, 785.5104980, -207.0206451, 838.7129517, -1032.7049561, 992.5311279
2: -198.3775940, 778.0156860, -211.4122925, 830.5660400, -1028.9434814, 989.4279785
3: -313.7880249, 816.0270386, -335.2691040, 871.3525391, -1185.1406250, 1151.2958984
4: -317.0524902, 778.5471191, -338.4065247, 830.8461914, -1147.8986816, 1116.9536133

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -175.7278595, 780.2539062, -918.1505127, 792.0089722
1: -172.6246338, 701.4801025, -219.7311096, 888.0158081, -1060.6403809, 921.2111816
2: -176.5765839, 694.5836182, -223.9974518, 879.2314453, -1055.8077393, 918.5810547
3: -279.6128540, 728.3481445, -355.7673950, 922.9997559, -1202.6125488, 1084.1153564
4: -282.3982544, 694.6760254, -358.6697693, 879.5055542, -1161.9036865, 1053.3458252

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980974, upper bound: 853.2963580
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984261, upper bound: 853.2965267
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2875704, upper bound: 853.2679952
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -175.7278595, 780.2539062, -935.1694946, 865.7015381
1: -193.9920044, 785.5104980, -219.7311096, 888.0158081, -1082.0075684, 1005.2415771
2: -198.3775940, 778.0156860, -223.9974518, 879.2314453, -1077.6087646, 1002.0131226
3: -313.7880249, 816.0270386, -355.7673950, 922.9997559, -1236.7878418, 1171.7941895
4: -317.0524902, 778.5471191, -358.6697693, 879.5055542, -1196.5578613, 1137.2167969

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974320, upper bound: 853.2959518
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974658, upper bound: 853.2960143
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2870757, upper bound: 853.2677586
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -157.9430389, 703.3171387, -848.1240234, 804.8965454
1: -181.2450409, 736.1990356, -197.7702942, 800.6807251, -981.9257812, 933.9692993
2: -185.1202850, 728.8364258, -202.2394409, 793.0823975, -978.2026367, 931.0758667
3: -293.8229980, 764.5282593, -319.9077759, 831.7758789, -1125.5987549, 1084.4356689
4: -296.4824219, 728.7455444, -323.1899719, 793.5729980, -1090.0552979, 1051.9355469

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -157.9430389, 703.3171387, -865.4146118, 880.5881348
1: -202.9634247, 822.4923096, -197.7702942, 800.6807251, -1003.6441040, 1020.2625732
2: -207.2748718, 814.4534302, -202.2394409, 793.0823975, -1000.3572998, 1016.6928711
3: -328.6972656, 854.5055542, -319.9077759, 831.7758789, -1160.4729004, 1174.4133301
4: -331.8298645, 814.7550659, -323.1899719, 793.5729980, -1125.4025879, 1137.9450684

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -171.4976959, 760.4559326, -905.2628174, 818.4512329
1: -181.2450409, 736.1990356, -214.4616547, 865.6424561, -1046.8874512, 950.6605835
2: -185.1202850, 728.8364258, -218.7769775, 857.2067871, -1042.3269043, 947.6134033
3: -293.8229980, 764.5282593, -346.8608398, 899.7705688, -1193.5935059, 1111.3890381
4: -296.4824219, 728.7455444, -349.8637695, 857.6834106, -1154.1657715, 1078.6092529

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008649, upper bound: 853.2990135
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010930, upper bound: 853.2993404
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -171.4976959, 760.4559326, -922.5534058, 894.1428223
1: -202.9634247, 822.4923096, -214.4616547, 865.6424561, -1068.6058350, 1036.9538574
2: -207.2748718, 814.4534302, -218.7769775, 857.2067871, -1064.4816895, 1033.2304688
3: -328.6972656, 854.5055542, -346.8608398, 899.7705688, -1228.4676514, 1201.3664551
4: -331.8298645, 814.7550659, -349.8637695, 857.6834106, -1189.5131836, 1164.6184082

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987433, upper bound: 853.2980528
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987360, upper bound: 853.2980902
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -170.9987030, 760.2313232, -157.9430389, 703.3171387, -874.3157959, 918.1742554
1: -213.8578033, 865.2389526, -197.7702942, 800.6807251, -1014.5384521, 1063.0092773
2: -218.0611115, 856.6461182, -202.2394409, 793.0823975, -1011.1434937, 1058.8854980
3: -346.3732300, 899.2537842, -319.9077759, 831.7758789, -1178.1489258, 1219.1612549
4: -349.3176575, 856.9051514, -323.1899719, 793.5729980, -1142.8906250, 1180.0950928

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2965267, upper bound: 853.2984261
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -170.9987030, 760.2313232, -171.4976959, 760.4559326, -931.4544678, 931.7289429
1: -213.8578033, 865.2389526, -214.4616547, 865.6424561, -1079.5002441, 1079.7005615
2: -218.0611115, 856.6461182, -218.7769775, 857.2067871, -1075.2679443, 1075.4229736
3: -346.3732300, 899.2537842, -346.8608398, 899.7705688, -1246.1437988, 1246.1145020
4: -349.3176575, 856.9051514, -349.8637695, 857.6834106, -1207.0010986, 1206.7687988

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -165.3467407, 736.9172363, -881.7241211, 812.3003540
1: -181.2450409, 736.1990356, -207.0206451, 838.7129517, -1019.9580078, 943.2196045
2: -185.1202850, 728.8364258, -211.4122925, 830.5660400, -1015.6861572, 940.2487183
3: -293.8229980, 764.5282593, -335.2691040, 871.3525391, -1165.1755371, 1099.7969971
4: -296.4824219, 728.7455444, -338.4065247, 830.8461914, -1127.3286133, 1067.1520996

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -165.3467407, 736.9172363, -899.0147095, 887.9919434
1: -202.9634247, 822.4923096, -207.0206451, 838.7129517, -1041.6763916, 1029.5129395
2: -207.2748718, 814.4534302, -211.4122925, 830.5660400, -1037.8408203, 1025.8657227
3: -328.6972656, 854.5055542, -335.2691040, 871.3525391, -1200.0498047, 1189.7746582
4: -331.8298645, 814.7550659, -338.4065247, 830.8461914, -1162.6760254, 1153.1613770

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -175.7278595, 780.2539062, -925.0607300, 822.6814575
1: -181.2450409, 736.1990356, -219.7311096, 888.0158081, -1069.2606201, 955.9300537
2: -185.1202850, 728.8364258, -223.9974518, 879.2314453, -1064.3514404, 952.8338623
3: -293.8229980, 764.5282593, -355.7673950, 922.9997559, -1216.8227539, 1120.2954102
4: -296.4824219, 728.7455444, -358.6697693, 879.5055542, -1175.9879150, 1087.4152832

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2978327, upper bound: 853.2954436
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981257, upper bound: 853.2957036
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2872398, upper bound: 853.2673767
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -175.7278595, 780.2539062, -942.3513184, 898.3730469
1: -202.9634247, 822.4923096, -219.7311096, 888.0158081, -1090.9790039, 1042.2232666
2: -207.2748718, 814.4534302, -223.9974518, 879.2314453, -1086.5062256, 1038.4509277
3: -328.6972656, 854.5055542, -355.7673950, 922.9997559, -1251.6968994, 1210.2729492
4: -331.8298645, 814.7550659, -358.6697693, 879.5055542, -1211.3352051, 1173.4245605

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961037, upper bound: 853.2947559
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2865769, upper bound: 853.2670831
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.34 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998366
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001635
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3002049, upper bound: 853.2993242
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2996511
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2998366, upper bound: 853.3011653
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3001635, upper bound: 853.3013934
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3010684, upper bound: 853.3010684
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2984261, upper bound: 853.2965267
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2875704, upper bound: 853.2679952
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2974658, upper bound: 853.2960143
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2870757, upper bound: 853.2677586
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3008649, upper bound: 853.2990135
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.3010930, upper bound: 853.2993404
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2987433, upper bound: 853.2980528
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2987360, upper bound: 853.2980902
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2965267, upper bound: 853.2984261
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2679908, upper bound: 853.2872354
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2981257, upper bound: 853.2957036
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2872398, upper bound: 853.2673767
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2961037, upper bound: 853.2947559
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -853.2865769, upper bound: 853.2670831

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -137.8967133, 616.2811279, -754.1777954, 754.1777954
1: -172.6246338, 701.4801025, -172.6246338, 701.4801025, -874.1047363, 874.1047363
2: -176.5765839, 694.5836182, -176.5765839, 694.5836182, -871.1602173, 871.1602173
3: -279.6128540, 728.3481445, -279.6128540, 728.3481445, -1007.9609985, 1007.9609985
4: -282.3982544, 694.6760254, -282.3982544, 694.6760254, -977.0742798, 977.0742798

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024182, upper bound: 853.3018986
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023290, upper bound: 853.3018811
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -154.9156036, 689.9736938, -827.8702393, 771.1967163
1: -172.6246338, 701.4801025, -193.9920044, 785.5104980, -958.1351318, 895.4721069
2: -176.5765839, 694.5836182, -198.3775940, 778.0156860, -954.5922852, 892.9611816
3: -279.6128540, 728.3481445, -313.7880249, 816.0270386, -1095.6398926, 1042.1362305
4: -282.3982544, 694.6760254, -317.0524902, 778.5471191, -1060.9450684, 1011.7285156

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024182, upper bound: 853.3018986
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023290, upper bound: 853.3018811
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -137.8967133, 616.2811279, -771.1967163, 827.8703003
1: -193.9920044, 785.5104980, -172.6246338, 701.4801025, -895.4721069, 958.1351318
2: -198.3775940, 778.0156860, -176.5765839, 694.5836182, -892.9611816, 954.5922852
3: -313.7880249, 816.0270386, -279.6128540, 728.3481445, -1042.1362305, 1095.6397705
4: -317.0524902, 778.5471191, -282.3982544, 694.6760254, -1011.7285156, 1060.9450684

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993337, upper bound: 853.3001869
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -154.9156036, 689.9736938, -844.8892822, 844.8892822
1: -193.9920044, 785.5104980, -193.9920044, 785.5104980, -979.5025024, 979.5025024
2: -198.3775940, 778.0156860, -198.3775940, 778.0156860, -976.3933105, 976.3933105
3: -313.7880249, 816.0270386, -313.7880249, 816.0270386, -1129.8150635, 1129.8150635
4: -317.0524902, 778.5471191, -317.0524902, 778.5471191, -1095.5994873, 1095.5993652

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993337, upper bound: 853.3001869
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -166.7426147, 740.2362061, -878.1328125, 783.0237427
1: -172.6246338, 701.4801025, -208.5561523, 842.6534424, -1015.2780151, 910.0361938
2: -176.5765839, 694.5836182, -212.7993317, 834.3977051, -1010.9743042, 907.3829346
3: -279.6128540, 728.3481445, -337.4005737, 875.8064575, -1155.4193115, 1065.7486572
4: -282.3982544, 694.6760254, -340.4402466, 834.8568115, -1117.2548828, 1035.1162109

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998361
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998366
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -174.2179718, 774.1820068, -912.0785522, 790.4990234
1: -172.6246338, 701.4801025, -217.9788666, 881.2302246, -1053.8547363, 919.4588623
2: -176.5765839, 694.5836182, -222.4038544, 872.6578369, -1049.2342529, 916.9874878
3: -279.6128540, 728.3481445, -352.7824707, 915.8316650, -1195.4443359, 1081.1306152
4: -282.3982544, 694.6760254, -355.6772156, 873.3172607, -1155.7153320, 1050.3532715

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001630
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001635
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -166.7426147, 740.2362061, -895.1517944, 856.7163086
1: -193.9920044, 785.5104980, -208.5561523, 842.6534424, -1036.6452637, 994.0666504
2: -198.3775940, 778.0156860, -212.7993317, 834.3977051, -1032.7751465, 990.8150024
3: -313.7880249, 816.0270386, -337.4005737, 875.8064575, -1189.5944824, 1153.4276123
4: -317.0524902, 778.5471191, -340.4402466, 834.8568115, -1151.9090576, 1118.9873047

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981699, upper bound: 853.2981424
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981699, upper bound: 853.2993242
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -174.2179718, 774.1820068, -929.0975952, 864.1915894
1: -193.9920044, 785.5104980, -217.9788666, 881.2302246, -1075.2220459, 1003.4893188
2: -198.3775940, 778.0156860, -222.4038544, 872.6578369, -1071.0352783, 1000.4195557
3: -313.7880249, 816.0270386, -352.7824707, 915.8316650, -1229.6195068, 1168.8094482
4: -317.0524902, 778.5471191, -355.6772156, 873.3172607, -1190.3695068, 1134.2243652

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983981, upper bound: 853.2984693
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983981, upper bound: 853.2996511
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -137.8967133, 616.2811279, -783.0237427, 878.1328125
1: -208.5561523, 842.6534424, -172.6246338, 701.4801025, -910.0361938, 1015.2780762
2: -212.7993317, 834.3977051, -176.5765839, 694.5836182, -907.3829346, 1010.9743042
3: -337.4005737, 875.8064575, -279.6128540, 728.3481445, -1065.7486572, 1155.4193115
4: -340.4402466, 834.8568115, -282.3982544, 694.6760254, -1035.1162109, 1117.2548828

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991035, upper bound: 853.3008319
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997421, upper bound: 853.3011354
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -154.9156036, 689.9736938, -856.7163086, 895.1517944
1: -208.5561523, 842.6534424, -193.9920044, 785.5104980, -994.0666504, 1036.6452637
2: -212.7993317, 834.3977051, -198.3775940, 778.0156860, -990.8150024, 1032.7751465
3: -337.4005737, 875.8064575, -313.7880249, 816.0270386, -1153.4276123, 1189.5944824
4: -340.4402466, 834.8568115, -317.0524902, 778.5471191, -1118.9873047, 1151.9090576

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2986981, upper bound: 853.3001666
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991859, upper bound: 853.3001751
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -137.8967133, 616.2811279, -790.4990234, 912.0785522
1: -217.9788666, 881.2302246, -172.6246338, 701.4801025, -919.4588623, 1053.8547363
2: -222.4038544, 872.6578369, -176.5765839, 694.5836182, -916.9874878, 1049.2343750
3: -352.7824707, 915.8316650, -279.6128540, 728.3481445, -1081.1306152, 1195.4442139
4: -355.6772156, 873.3172607, -282.3982544, 694.6760254, -1050.3532715, 1155.7154541

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993811, upper bound: 853.3010550
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000690, upper bound: 853.3013525
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -154.9156036, 689.9736938, -864.1915894, 929.0975952
1: -217.9788666, 881.2302246, -193.9920044, 785.5104980, -1003.4893188, 1075.2220459
2: -222.4038544, 872.6578369, -198.3775940, 778.0156860, -1000.4195557, 1071.0351562
3: -352.7824707, 915.8316650, -313.7880249, 816.0270386, -1168.8094482, 1229.6195068
4: -355.6772156, 873.3172607, -317.0524902, 778.5471191, -1134.2243652, 1190.3695068

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2989757, upper bound: 853.3003897
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995128, upper bound: 853.3003921
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -166.7426147, 740.2362061, -906.9788208, 906.9788208
1: -208.5561523, 842.6534424, -208.5561523, 842.6534424, -1051.2095947, 1051.2095947
2: -212.7993317, 834.3977051, -212.7993317, 834.3977051, -1047.1970215, 1047.1970215
3: -337.4005737, 875.8064575, -337.4005737, 875.8064575, -1213.2070312, 1213.2070312
4: -340.4402466, 834.8568115, -340.4402466, 834.8568115, -1175.2969971, 1175.2969971

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015195, upper bound: 853.3008957
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023677, upper bound: 853.3015926
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -174.2179718, 774.1820068, -940.9246216, 914.4541016
1: -208.5561523, 842.6534424, -217.9788666, 881.2302246, -1089.7863770, 1060.6322021
2: -212.7993317, 834.3977051, -222.4038544, 872.6578369, -1085.4571533, 1056.8015137
3: -337.4005737, 875.8064575, -352.7824707, 915.8316650, -1253.2320557, 1228.5888672
4: -340.4402466, 834.8568115, -355.6772156, 873.3172607, -1213.7575684, 1190.5340576

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015195, upper bound: 853.3008957
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023677, upper bound: 853.3015926
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -166.7426147, 740.2362061, -914.4541016, 940.9245605
1: -217.9788666, 881.2302246, -208.5561523, 842.6534424, -1060.6322021, 1089.7863770
2: -222.4038544, 872.6578369, -212.7993317, 834.3977051, -1056.8013916, 1085.4571533
3: -352.7824707, 915.8316650, -337.4005737, 875.8064575, -1228.5888672, 1253.2320557
4: -355.6772156, 873.3172607, -340.4402466, 834.8568115, -1190.5340576, 1213.7575684

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006488, upper bound: 853.3007492
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010294, upper bound: 853.3010294
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -174.2179718, 774.1820068, -948.3998413, 948.3998413
1: -217.9788666, 881.2302246, -217.9788666, 881.2302246, -1099.2089844, 1099.2089844
2: -222.4038544, 872.6578369, -222.4038544, 872.6578369, -1095.0616455, 1095.0616455
3: -352.7824707, 915.8316650, -352.7824707, 915.8316650, -1268.6138916, 1268.6138916
4: -355.6772156, 873.3172607, -355.6772156, 873.3172607, -1228.9945068, 1228.9945068

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006488, upper bound: 853.3007492
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010294, upper bound: 853.3010294
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -144.8068848, 646.9536133, -784.8502197, 761.0880127
1: -172.6246338, 701.4801025, -181.2450409, 736.1990356, -908.8236694, 882.7251587
2: -176.5765839, 694.5836182, -185.1202850, 728.8364258, -905.4130249, 879.7038574
3: -279.6128540, 728.3481445, -293.8229980, 764.5282593, -1044.1411133, 1022.1710815
4: -282.3982544, 694.6760254, -296.4824219, 728.7455444, -1011.1437988, 991.1584473

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011597, upper bound: 853.3005306
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010706, upper bound: 853.3005028
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -162.0976410, 722.6452026, -860.5418091, 778.3786011
1: -172.6246338, 701.4801025, -202.9634247, 822.4923096, -995.1169434, 904.4434814
2: -176.5765839, 694.5836182, -207.2748718, 814.4534302, -991.0300293, 901.8585205
3: -279.6128540, 728.3481445, -328.6972656, 854.5055542, -1134.1184082, 1057.0451660
4: -282.3982544, 694.6760254, -331.8298645, 814.7550659, -1097.1529541, 1026.5058594

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011597, upper bound: 853.3005306
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010706, upper bound: 853.3005028
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -144.8068848, 646.9536133, -801.8692017, 834.7805786
1: -193.9920044, 785.5104980, -181.2450409, 736.1990356, -930.1910400, 966.7555542
2: -198.3775940, 778.0156860, -185.1202850, 728.8364258, -927.2139893, 963.1359863
3: -313.7880249, 816.0270386, -293.8229980, 764.5282593, -1078.3162842, 1109.8498535
4: -317.0524902, 778.5471191, -296.4824219, 728.7455444, -1045.7978516, 1075.0294189

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980753, upper bound: 853.2987806
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -162.0976410, 722.6452026, -877.5607910, 852.0712280
1: -193.9920044, 785.5104980, -202.9634247, 822.4923096, -1016.4843140, 988.4739380
2: -198.3775940, 778.0156860, -207.2748718, 814.4534302, -1012.8309326, 985.2905273
3: -313.7880249, 816.0270386, -328.6972656, 854.5055542, -1168.2935791, 1144.7241211
4: -317.0524902, 778.5471191, -331.8298645, 814.7550659, -1131.8071289, 1110.3768311

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980753, upper bound: 853.2987806
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -170.9987030, 760.2313232, -898.1279297, 787.2797241
1: -172.6246338, 701.4801025, -213.8578033, 865.2389526, -1037.8635254, 915.3378296
2: -176.5765839, 694.5836182, -218.0611115, 856.6461182, -1033.2225342, 912.6447144
3: -279.6128540, 728.3481445, -346.3732300, 899.2537842, -1178.8665771, 1074.7211914
4: -282.3982544, 694.6760254, -349.3176575, 856.9051514, -1139.3031006, 1043.9936523

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2875704, upper bound: 853.2679952
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2875704, upper bound: 853.2679952
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -170.9987030, 760.2313232, -915.1469116, 860.9723511
1: -193.9920044, 785.5104980, -213.8578033, 865.2389526, -1059.2309570, 999.3682861
2: -198.3775940, 778.0156860, -218.0611115, 856.6461182, -1055.0235596, 996.0767822
3: -313.7880249, 816.0270386, -346.3732300, 899.2537842, -1213.0417480, 1162.4001465
4: -317.0524902, 778.5471191, -349.3176575, 856.9051514, -1173.9573975, 1127.8647461

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2860736, upper bound: 853.2674942
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2860736, upper bound: 853.2677586
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -137.8967133, 616.2811279, -761.0880127, 784.8502197
1: -181.2450409, 736.1990356, -172.6246338, 701.4801025, -882.7251587, 908.8236084
2: -185.1202850, 728.8364258, -176.5765839, 694.5836182, -879.7039185, 905.4130249
3: -293.8229980, 764.5282593, -279.6128540, 728.3481445, -1022.1710815, 1044.1411133
4: -296.4824219, 728.7455444, -282.3982544, 694.6760254, -991.1584473, 1011.1437988

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021998, upper bound: 853.3014932
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017310, upper bound: 853.3008221
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -154.9156036, 689.9736938, -834.7805176, 801.8692017
1: -181.2450409, 736.1990356, -193.9920044, 785.5104980, -966.7555542, 930.1910400
2: -185.1202850, 728.8364258, -198.3775940, 778.0156860, -963.1359863, 927.2139893
3: -293.8229980, 764.5282593, -313.7880249, 816.0270386, -1109.8498535, 1078.3162842
4: -296.4824219, 728.7455444, -317.0524902, 778.5471191, -1075.0294189, 1045.7978516

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021998, upper bound: 853.3014932
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017310, upper bound: 853.3008221
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -137.8967133, 616.2811279, -778.3786011, 860.5418091
1: -202.9634247, 822.4923096, -172.6246338, 701.4801025, -904.4434814, 995.1169434
2: -207.2748718, 814.4534302, -176.5765839, 694.5836182, -901.8585205, 991.0299683
3: -328.6972656, 854.5055542, -279.6128540, 728.3481445, -1057.0451660, 1134.1184082
4: -331.8298645, 814.7550659, -282.3982544, 694.6760254, -1026.5057373, 1097.1529541

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3000719
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2986716, upper bound: 853.2992370
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -154.9156036, 689.9736938, -852.0712280, 877.5607910
1: -202.9634247, 822.4923096, -193.9920044, 785.5104980, -988.4739380, 1016.4843140
2: -207.2748718, 814.4534302, -198.3775940, 778.0156860, -985.2905273, 1012.8309326
3: -328.6972656, 854.5055542, -313.7880249, 816.0270386, -1144.7239990, 1168.2935791
4: -331.8298645, 814.7550659, -317.0524902, 778.5471191, -1110.3768311, 1131.8071289

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3000719
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2986716, upper bound: 853.2992370
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -166.7426147, 740.2362061, -885.0430908, 813.6962280
1: -181.2450409, 736.1990356, -208.5561523, 842.6534424, -1023.8984985, 944.7550049
2: -185.1202850, 728.8364258, -212.7993317, 834.3977051, -1019.5179443, 941.6357422
3: -293.8229980, 764.5282593, -337.4005737, 875.8064575, -1169.6293945, 1101.9288330
4: -296.4824219, 728.7455444, -340.4402466, 834.8568115, -1131.3391113, 1069.1857910

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008649, upper bound: 853.2990135
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008649, upper bound: 853.2990135
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -174.2179718, 774.1820068, -918.9888306, 821.1715088
1: -181.2450409, 736.1990356, -217.9788666, 881.2302246, -1062.4752197, 954.1777344
2: -185.1202850, 728.8364258, -222.4038544, 872.6578369, -1057.7779541, 951.2402954
3: -293.8229980, 764.5282593, -352.7824707, 915.8316650, -1209.6542969, 1117.3106689
4: -296.4824219, 728.7455444, -355.6772156, 873.3172607, -1169.7995605, 1084.4227295

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010930, upper bound: 853.2993404
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010930, upper bound: 853.2993404
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -166.7426147, 740.2362061, -902.3337402, 889.3878174
1: -202.9634247, 822.4923096, -208.5561523, 842.6534424, -1045.6168213, 1031.0484619
2: -207.2748718, 814.4534302, -212.7993317, 834.3977051, -1041.6726074, 1027.2528076
3: -328.6972656, 854.5055542, -337.4005737, 875.8064575, -1204.5035400, 1191.9061279
4: -331.8298645, 814.7550659, -340.4402466, 834.8568115, -1166.6864014, 1155.1950684

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975078, upper bound: 853.2971925
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975078, upper bound: 853.2980528
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -174.2179718, 774.1820068, -936.2794189, 896.8630981
1: -202.9634247, 822.4923096, -217.9788666, 881.2302246, -1084.1936035, 1040.4710693
2: -207.2748718, 814.4534302, -222.4038544, 872.6578369, -1079.9327393, 1036.8572998
3: -328.6972656, 854.5055542, -352.7824707, 915.8316650, -1244.5284424, 1207.2880859
4: -331.8298645, 814.7550659, -355.6772156, 873.3172607, -1205.1468506, 1170.4320068

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977360, upper bound: 853.2975194
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977360, upper bound: 853.2980902
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -170.9987030, 760.2313232, -137.8967133, 616.2811279, -787.2797852, 898.1279297
1: -213.8578033, 865.2389526, -172.6246338, 701.4801025, -915.3378296, 1037.8635254
2: -218.0611115, 856.6461182, -176.5765839, 694.5836182, -912.6447144, 1033.2225342
3: -346.3732300, 899.2537842, -279.6128540, 728.3481445, -1074.7213135, 1178.8665771
4: -349.3176575, 856.9051514, -282.3982544, 694.6760254, -1043.9936523, 1139.3031006

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2963572, upper bound: 853.2980928
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -170.9987030, 760.2313232, -154.9156036, 689.9736938, -860.9722900, 915.1469116
1: -213.8578033, 865.2389526, -193.9920044, 785.5104980, -999.3682861, 1059.2308350
2: -218.0611115, 856.6461182, -198.3775940, 778.0156860, -996.0767822, 1055.0235596
3: -346.3732300, 899.2537842, -313.7880249, 816.0270386, -1162.4001465, 1213.0417480
4: -349.3176575, 856.9051514, -317.0524902, 778.5471191, -1127.8647461, 1173.9573975

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2959518, upper bound: 853.2974274
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -144.8068848, 646.9536133, -791.7604980, 791.7604980
1: -181.2450409, 736.1990356, -181.2450409, 736.1990356, -917.4440308, 917.4440308
2: -185.1202850, 728.8364258, -185.1202850, 728.8364258, -913.9566650, 913.9566650
3: -293.8229980, 764.5282593, -293.8229980, 764.5282593, -1058.3510742, 1058.3510742
4: -296.4824219, 728.7455444, -296.4824219, 728.7455444, -1025.2279053, 1025.2280273

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009419, upper bound: 853.3001302
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996960
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -162.0976410, 722.6452026, -867.4520874, 809.0511475
1: -181.2450409, 736.1990356, -202.9634247, 822.4923096, -1003.7373657, 939.1622925
2: -185.1202850, 728.8364258, -207.2748718, 814.4534302, -999.5736084, 936.1113281
3: -293.8229980, 764.5282593, -328.6972656, 854.5055542, -1148.3286133, 1093.2253418
4: -296.4824219, 728.7455444, -331.8298645, 814.7550659, -1111.2371826, 1060.5753174

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009419, upper bound: 853.3001302
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996960
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -144.8068848, 646.9536133, -809.0511475, 867.4520874
1: -202.9634247, 822.4923096, -181.2450409, 736.1990356, -939.1623535, 1003.7373657
2: -207.2748718, 814.4534302, -185.1202850, 728.8364258, -936.1113281, 999.5736084
3: -328.6972656, 854.5055542, -293.8229980, 764.5282593, -1093.2253418, 1148.3286133
4: -331.8298645, 814.7550659, -296.4824219, 728.7455444, -1060.5753174, 1111.2371826

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974132, upper bound: 853.2978749
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -162.0976410, 722.6452026, -884.7427368, 884.7427368
1: -202.9634247, 822.4923096, -202.9634247, 822.4923096, -1025.4555664, 1025.4555664
2: -207.2748718, 814.4534302, -207.2748718, 814.4534302, -1021.7282715, 1021.7282715
3: -328.6972656, 854.5055542, -328.6972656, 854.5055542, -1183.2028809, 1183.2028809
4: -331.8298645, 814.7550659, -331.8298645, 814.7550659, -1146.5844727, 1146.5844727

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974132, upper bound: 853.2978749
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -170.9987030, 760.2313232, -905.0382080, 817.9522095
1: -181.2450409, 736.1990356, -213.8578033, 865.2389526, -1046.4840088, 950.0567017
2: -185.1202850, 728.8364258, -218.0611115, 856.6461182, -1041.7662354, 946.8975220
3: -293.8229980, 764.5282593, -346.3732300, 899.2537842, -1193.0766602, 1110.9013672
4: -296.4824219, 728.7455444, -349.3176575, 856.9051514, -1153.3874512, 1078.0632324

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2872398, upper bound: 853.2673767
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2872398, upper bound: 853.2673767
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -170.9987030, 760.2313232, -922.3288574, 893.6438599
1: -202.9634247, 822.4923096, -213.8578033, 865.2389526, -1068.2023926, 1036.3500977
2: -207.2748718, 814.4534302, -218.0611115, 856.6461182, -1063.9210205, 1032.5145264
3: -328.6972656, 854.5055542, -346.3732300, 899.2537842, -1227.9506836, 1200.8787842
4: -331.8298645, 814.7550659, -349.3176575, 856.9051514, -1188.7347412, 1164.0723877

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2854116, upper bound: 853.2665496
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2854116, upper bound: 853.2670831
time: 0.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.55 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3024182, upper bound: 853.3018986
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3023290, upper bound: 853.3018811
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3024182, upper bound: 853.3018986
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3023290, upper bound: 853.3018811
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2993337, upper bound: 853.3001869
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2993337, upper bound: 853.3001869
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3013687, upper bound: 853.3013687
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998361
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998366
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001630
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001635
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2981699, upper bound: 853.2981424
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2981699, upper bound: 853.2993242
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2983981, upper bound: 853.2984693
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2983981, upper bound: 853.2996511
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2993242, upper bound: 853.3002049
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2996511, upper bound: 853.3004331
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3015195, upper bound: 853.3008957
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3023677, upper bound: 853.3015926
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3015195, upper bound: 853.3008957
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3023677, upper bound: 853.3015926
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3006488, upper bound: 853.3007492
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3010294, upper bound: 853.3010294
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3006488, upper bound: 853.3007492
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3010294, upper bound: 853.3010294
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3011597, upper bound: 853.3005306
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3010706, upper bound: 853.3005028
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3011597, upper bound: 853.3005306
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3010706, upper bound: 853.3005028
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2980753, upper bound: 853.2987806
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2980753, upper bound: 853.2987806
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3001103, upper bound: 853.3000067
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2875704, upper bound: 853.2679952
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2875704, upper bound: 853.2679952
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2860736, upper bound: 853.2674942
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2860736, upper bound: 853.2677586
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3021998, upper bound: 853.3014932
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3017310, upper bound: 853.3008221
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3021998, upper bound: 853.3014932
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3017310, upper bound: 853.3008221
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2986716, upper bound: 853.2992370
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2986716, upper bound: 853.2992370
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3000067, upper bound: 853.3001103
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3008649, upper bound: 853.2990135
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3008649, upper bound: 853.2990135
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3010930, upper bound: 853.2993404
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3010930, upper bound: 853.2993404
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2975078, upper bound: 853.2971925
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2975078, upper bound: 853.2980528
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2977360, upper bound: 853.2975194
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2977360, upper bound: 853.2980902
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2960143, upper bound: 853.2974658
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3009419, upper bound: 853.3001302
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996960
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3009419, upper bound: 853.3001302
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996960
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2974132, upper bound: 853.2978749
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2974132, upper bound: 853.2978749
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2987483, upper bound: 853.2987483
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2872398, upper bound: 853.2673767
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2872398, upper bound: 853.2673767
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2854116, upper bound: 853.2665496
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.55
Output dim: 0, lower bound: -853.2854116, upper bound: 853.2670831

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -134.0090179, 599.6547241, -137.8967133, 616.2811279, -750.2900391, 737.5513916
1: -167.7749939, 682.5620117, -172.6246338, 701.4801025, -869.2551270, 855.1866455
2: -171.6572723, 675.8182983, -176.5765839, 694.5836182, -866.2409058, 852.3948975
3: -271.8343506, 708.6510620, -279.6128540, 728.3481445, -1000.1824951, 988.2639160
4: -274.6115723, 675.8950195, -282.3982544, 694.6760254, -969.2875977, 958.2932739

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028410, upper bound: 853.3028410
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028410, upper bound: 853.3028410
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -141.2080231, 632.3185425, -137.8967133, 616.2811279, -757.4891357, 770.2152100
1: -176.8826447, 719.6705933, -172.6246338, 701.4801025, -878.3627319, 892.2952271
2: -180.9474182, 712.7294922, -176.5765839, 694.5836182, -875.5310059, 889.3060913
3: -286.6737976, 747.1620483, -279.6128540, 728.3481445, -1015.0219727, 1026.7749023
4: -289.2762451, 713.0266724, -282.3982544, 694.6760254, -983.9522705, 995.4249268

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028410, upper bound: 853.3028415
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028410, upper bound: 853.3028415
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -134.0090179, 599.6547241, -154.9156036, 689.9736938, -823.9826050, 754.5703125
1: -167.7749939, 682.5620117, -193.9920044, 785.5104980, -953.2855225, 876.5540161
2: -171.6572723, 675.8182983, -198.3775940, 778.0156860, -949.6729736, 874.1958618
3: -271.8343506, 708.6510620, -313.7880249, 816.0270386, -1087.8613281, 1022.4390869
4: -274.6115723, 675.8950195, -317.0524902, 778.5471191, -1053.1585693, 992.9475098

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011472, upper bound: 853.2998456
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011472, upper bound: 853.3018806
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -141.2080231, 632.3185425, -154.9156036, 689.9736938, -831.1815796, 787.2341309
1: -176.8826447, 719.6705933, -193.9920044, 785.5104980, -962.3931274, 913.6625977
2: -180.9474182, 712.7294922, -198.3775940, 778.0156860, -958.9631348, 911.1069946
3: -286.6737976, 747.1620483, -313.7880249, 816.0270386, -1102.7008057, 1060.9500732
4: -289.2762451, 713.0266724, -317.0524902, 778.5471191, -1067.8232422, 1030.0789795

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011472, upper bound: 853.2998461
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011472, upper bound: 853.3018811
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -151.6145020, 675.7196045, -137.8967133, 616.2811279, -767.8955688, 813.6162720
1: -189.8713989, 769.2893066, -172.6246338, 701.4801025, -891.3515015, 941.9139404
2: -194.1967163, 761.9385376, -176.5765839, 694.5836182, -888.7802124, 938.5151367
3: -307.1603394, 799.1470337, -279.6128540, 728.3481445, -1035.5084229, 1078.7598877
4: -310.4094238, 762.4762573, -282.3982544, 694.6760254, -1005.0854492, 1044.8742676

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998456, upper bound: 853.3011472
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998456, upper bound: 853.3011472
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -157.3274384, 702.2606201, -137.8967133, 616.2811279, -773.6085815, 840.1571655
1: -197.1456146, 799.4273071, -172.6246338, 701.4801025, -898.6256714, 972.0519409
2: -201.6119537, 791.9401245, -176.5765839, 694.5836182, -896.1954956, 968.5167236
3: -319.0841064, 830.3767090, -279.6128540, 728.3481445, -1047.4322510, 1109.9895020
4: -322.1704712, 792.6814575, -282.3982544, 694.6760254, -1016.8464966, 1075.0797119

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018806, upper bound: 853.3023290
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018806, upper bound: 853.3023290
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -151.6145020, 675.7196045, -154.9156036, 689.9736938, -841.5881348, 830.6351929
1: -189.8713989, 769.2893066, -193.9920044, 785.5104980, -975.3818970, 963.2813110
2: -194.1967163, 761.9385376, -198.3775940, 778.0156860, -972.2123413, 960.3161621
3: -307.1603394, 799.1470337, -313.7880249, 816.0270386, -1123.1873779, 1112.9350586
4: -310.4094238, 762.4762573, -317.0524902, 778.5471191, -1088.9565430, 1079.5284424

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981519, upper bound: 853.2981519
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981519, upper bound: 853.3001869
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -157.3274384, 702.2606201, -154.9156036, 689.9736938, -847.3011475, 857.1762085
1: -197.1456146, 799.4273071, -193.9920044, 785.5104980, -982.6561279, 993.4193115
2: -201.6119537, 791.9401245, -198.3775940, 778.0156860, -979.6276245, 990.3176880
3: -319.0841064, 830.3767090, -313.7880249, 816.0270386, -1135.1110840, 1144.1647949
4: -322.1704712, 792.6814575, -317.0524902, 778.5471191, -1100.7175293, 1109.7338867

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001869, upper bound: 853.2993337
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001869, upper bound: 853.3013687
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -134.0090179, 599.6547241, -166.7426147, 740.2362061, -874.2451172, 766.3973389
1: -167.7749939, 682.5620117, -208.5561523, 842.6534424, -1010.4284058, 891.1181030
2: -171.6572723, 675.8182983, -212.7993317, 834.3977051, -1006.0549927, 888.6176147
3: -271.8343506, 708.6510620, -337.4005737, 875.8064575, -1147.6408691, 1046.0515137
4: -274.6115723, 675.8950195, -340.4402466, 834.8568115, -1109.4682617, 1016.3352661

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.2983133
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011354, upper bound: 853.2997421
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998361
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011653, upper bound: 853.2998361
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -141.2080231, 632.3185425, -166.7426147, 740.2362061, -881.4442139, 799.0611572
1: -176.8826447, 719.6705933, -208.5561523, 842.6534424, -1019.5360107, 928.2267456
2: -180.9474182, 712.7294922, -212.7993317, 834.3977051, -1015.3450928, 925.5288086
3: -286.6737976, 747.1620483, -337.4005737, 875.8064575, -1162.4802246, 1084.5626221
4: -289.2762451, 713.0266724, -340.4402466, 834.8568115, -1124.1329346, 1053.4669189

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.2991035
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011354, upper bound: 853.2997421
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025811, upper bound: 853.3028234
time: 0.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -853.3025811, upper bound: 853.3028234
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -182.6434631, 808.2009888, -987.4398193, 976.6322021
1: -224.1477814, 903.8054199, -228.3799133, 920.0372925, -1144.1850586, 1132.1853027
2: -228.6609802, 895.1587524, -232.9678345, 911.1979980, -1139.8587646, 1128.1265869
3: -362.4073792, 939.4089966, -369.0971680, 956.3012085, -1318.7084961, 1308.5061035
4: -365.4209595, 895.6079712, -372.2211304, 911.6491089, -1277.0700684, 1267.8291016

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -183.7534332, 814.8997803, -182.0839081, 805.9441528, -989.6975708, 996.9837036
1: -229.7754364, 927.4383545, -227.6815796, 917.4570923, -1147.2325439, 1155.1197510
2: -234.2279053, 918.4588013, -232.2557373, 908.6355591, -1142.8635254, 1150.7145996
3: -371.8710327, 963.9637451, -368.0082703, 953.6092529, -1325.4802246, 1331.9720459
4: -374.7720337, 918.7300415, -371.1185303, 909.0764771, -1283.8485107, 1289.8486328

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
time: 0.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -179.2388916, 793.9887695, -973.2276001, 973.2276001
1: -224.1477814, 903.8054199, -224.1477814, 903.8054199, -1127.9532471, 1127.9532471
2: -228.6609802, 895.1587524, -228.6609802, 895.1587524, -1123.8197021, 1123.8197021
3: -362.4073792, 939.4089966, -362.4073792, 939.4089966, -1301.8164062, 1301.8164062
4: -365.4209595, 895.6079712, -365.4209595, 895.6079712, -1261.0289307, 1261.0289307

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -183.7534332, 814.8997803, -994.1386108, 977.7421265
1: -224.1477814, 903.8054199, -229.7754364, 927.4383545, -1151.5861816, 1133.5808105
2: -228.6609802, 895.1587524, -234.2279053, 918.4588013, -1147.1196289, 1129.3867188
3: -362.4073792, 939.4089966, -371.8710327, 963.9637451, -1326.3710938, 1311.2800293
4: -365.4209595, 895.6079712, -374.7720337, 918.7300415, -1284.1510010, 1270.3800049

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -182.7627106, 810.6425171, -160.7026215, 714.6928101, -897.4555054, 971.3450928
1: -228.5412750, 922.5975952, -201.2071228, 813.6656494, -1042.2069092, 1123.8045654
2: -232.9844208, 913.6407471, -205.7275391, 805.9682617, -1038.9525146, 1119.3682861
3: -369.8891296, 958.9255371, -325.3055725, 845.2979126, -1215.1868896, 1284.2310791
4: -372.8100586, 913.8994751, -328.6417847, 806.4979858, -1179.3079834, 1242.5412598

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -182.6057129, 809.9552002, -174.1690826, 771.6924438, -954.2980347, 984.1242676
1: -228.3437958, 921.8109741, -217.7790070, 878.4711914, -1106.8149414, 1139.5899658
2: -232.7684479, 912.8629761, -222.1530457, 869.8740234, -1102.6424561, 1135.0159912
3: -369.5759583, 958.1173706, -352.1272278, 913.1160889, -1282.6920166, 1310.2446289
4: -372.4694519, 913.1493530, -355.2000427, 870.3616943, -1242.8309326, 1268.3493652

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.60 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -178.1380463, 789.2713623, -947.2142944, 881.4552002
1: -197.7702942, 800.6807251, -222.7793884, 898.4378052, -1096.2078857, 1023.4600830
2: -202.2394409, 793.0823975, -227.2813721, 889.8236084, -1092.0628662, 1020.3637695
3: -319.9077759, 831.7758789, -360.2123108, 933.8228149, -1253.7304688, 1191.9879150
4: -323.1899719, 793.5729980, -363.2372131, 890.2650757, -1213.4550781, 1156.8099365

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -178.1753540, 789.3782959, -960.8759155, 938.6312256
1: -214.4616547, 865.6424561, -222.8191376, 898.5571899, -1113.0187988, 1088.4615479
2: -218.7769775, 857.2067871, -227.3043976, 889.9409790, -1108.7177734, 1084.5109863
3: -346.8608398, 899.7705688, -360.2748718, 933.9605713, -1280.8214111, 1260.0454102
4: -349.8637695, 857.6834106, -363.2850342, 890.4031372, -1240.2666016, 1220.9685059

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -182.7627106, 810.6425171, -968.5854492, 886.0798340
1: -197.7702942, 800.6807251, -228.5412750, 922.5975952, -1120.3676758, 1029.2220459
2: -202.2394409, 793.0823975, -232.9844208, 913.6407471, -1115.8801270, 1026.0667725
3: -319.9077759, 831.7758789, -369.8891296, 958.9255371, -1278.8331299, 1201.6647949
4: -323.1899719, 793.5729980, -372.8100586, 913.8994751, -1237.0894775, 1166.3829346

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -182.6057129, 809.9552002, -981.4528809, 943.0615234
1: -214.4616547, 865.6424561, -228.3437958, 921.8109741, -1136.2725830, 1093.9862061
2: -218.7769775, 857.2067871, -232.7684479, 912.8629761, -1131.6398926, 1089.9752197
3: -346.8608398, 899.7705688, -369.5759583, 958.1173706, -1304.9781494, 1269.3465576
4: -349.8637695, 857.6834106, -372.4694519, 913.1493530, -1263.0129395, 1230.1528320

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -160.7026215, 714.6928101, -880.0395508, 897.6198120
1: -207.0206451, 838.7129517, -201.2071228, 813.6656494, -1020.6862183, 1039.9199219
2: -211.4122925, 830.5660400, -205.7275391, 805.9682617, -1017.3805542, 1036.2935791
3: -335.2691040, 871.3525391, -325.3055725, 845.2979126, -1180.5667725, 1196.6580811
4: -338.4065247, 830.8461914, -328.6417847, 806.4979858, -1144.9045410, 1159.4880371

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -160.7026215, 714.6928101, -890.4206543, 940.9564209
1: -219.7311096, 888.0158081, -201.2071228, 813.6656494, -1033.3967285, 1089.2226562
2: -223.9974518, 879.2314453, -205.7275391, 805.9682617, -1029.9656982, 1084.9589844
3: -355.7673950, 922.9997559, -325.3055725, 845.2979126, -1201.0650635, 1248.3052979
4: -358.6697693, 879.5055542, -328.6417847, 806.4979858, -1165.1677246, 1208.1473389

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -174.1690826, 771.6924438, -937.0391846, 911.0863037
1: -207.0206451, 838.7129517, -217.7790070, 878.4711914, -1085.4918213, 1056.4919434
2: -211.4122925, 830.5660400, -222.1530457, 869.8740234, -1081.2863770, 1052.7191162
3: -335.2691040, 871.3525391, -352.1272278, 913.1160889, -1248.3852539, 1223.4797363
4: -338.4065247, 830.8461914, -355.2000427, 870.3616943, -1208.7679443, 1186.0462646

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -174.1690826, 771.6924438, -947.4202881, 954.4229126
1: -219.7311096, 888.0158081, -217.7790070, 878.4711914, -1098.2022705, 1105.7947998
2: -223.9974518, 879.2314453, -222.1530457, 869.8740234, -1093.8714600, 1101.3845215
3: -355.7673950, 922.9997559, -352.1272278, 913.1160889, -1268.8835449, 1275.1269531
4: -358.6697693, 879.5055542, -355.2000427, 870.3616943, -1229.0311279, 1234.7055664

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.67 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3030941, upper bound: 853.3030941
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3024013, upper bound: 853.3025225
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3020061, upper bound: 853.3021835
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -853.3018296, upper bound: 853.3018296

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -157.9430389, 703.3171387, -861.2600708, 861.2600708
1: -197.7702942, 800.6807251, -197.7702942, 800.6807251, -998.4510498, 998.4510498
2: -202.2394409, 793.0823975, -202.2394409, 793.0823975, -995.3218384, 995.3218384
3: -319.9077759, 831.7758789, -319.9077759, 831.7758789, -1151.6832275, 1151.6833496
4: -323.1899719, 793.5729980, -323.1899719, 793.5729980, -1116.7629395, 1116.7629395

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014599, upper bound: 853.3002908
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2995889
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -171.4976959, 760.4559326, -918.3988647, 874.8148193
1: -197.7702942, 800.6807251, -214.4616547, 865.6424561, -1063.4125977, 1015.1423340
2: -202.2394409, 793.0823975, -218.7769775, 857.2067871, -1059.4461670, 1011.8593750
3: -319.9077759, 831.7758789, -346.8608398, 899.7705688, -1219.6783447, 1178.6365967
4: -323.1899719, 793.5729980, -349.8637695, 857.6834106, -1180.8734131, 1143.4365234

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014599, upper bound: 853.3002908
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2995889
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -157.9430389, 703.3171387, -874.8147583, 918.3988647
1: -214.4616547, 865.6424561, -197.7702942, 800.6807251, -1015.1423340, 1063.4125977
2: -218.7769775, 857.2067871, -202.2394409, 793.0823975, -1011.8593750, 1059.4461670
3: -346.8608398, 899.7705688, -319.9077759, 831.7758789, -1178.6365967, 1219.6783447
4: -349.8637695, 857.6834106, -323.1899719, 793.5729980, -1143.4365234, 1180.8734131

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023592, upper bound: 853.3016342
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -171.4976959, 760.4559326, -931.9534912, 931.9534912
1: -214.4616547, 865.6424561, -214.4616547, 865.6424561, -1080.1040039, 1080.1041260
2: -218.7769775, 857.2067871, -218.7769775, 857.2067871, -1075.9836426, 1075.9836426
3: -346.8608398, 899.7705688, -346.8608398, 899.7705688, -1246.6313477, 1246.6313477
4: -349.8637695, 857.6834106, -349.8637695, 857.6834106, -1207.5471191, 1207.5471191

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023592, upper bound: 853.3016342
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -165.3467407, 736.9172363, -894.8601685, 868.6638794
1: -197.7702942, 800.6807251, -207.0206451, 838.7129517, -1036.4832764, 1007.7013550
2: -202.2394409, 793.0823975, -211.4122925, 830.5660400, -1032.8054199, 1004.4946899
3: -319.9077759, 831.7758789, -335.2691040, 871.3525391, -1191.2602539, 1167.0446777
4: -323.1899719, 793.5729980, -338.4065247, 830.8461914, -1154.0361328, 1131.9794922

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979258, upper bound: 853.2962687
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -157.9430389, 703.3171387, -175.7278595, 780.2539062, -938.1967773, 879.0449829
1: -197.7702942, 800.6807251, -219.7311096, 888.0158081, -1085.7858887, 1020.4118042
2: -202.2394409, 793.0823975, -223.9974518, 879.2314453, -1081.4708252, 1017.0798340
3: -319.9077759, 831.7758789, -355.7673950, 922.9997559, -1242.9074707, 1187.5429688
4: -323.1899719, 793.5729980, -358.6697693, 879.5055542, -1202.6955566, 1152.2426758

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979258, upper bound: 853.2962687
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -165.3467407, 736.9172363, -908.4148560, 925.8026733
1: -214.4616547, 865.6424561, -207.0206451, 838.7129517, -1053.1745605, 1072.6630859
2: -218.7769775, 857.2067871, -211.4122925, 830.5660400, -1049.3428955, 1068.6191406
3: -346.8608398, 899.7705688, -335.2691040, 871.3525391, -1218.2133789, 1235.0395508
4: -349.8637695, 857.6834106, -338.4065247, 830.8461914, -1180.7098389, 1196.0899658

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2809122, upper bound: 853.2643276
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2793620, upper bound: 853.2634073
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -171.4976959, 760.4559326, -175.7278595, 780.2539062, -951.7514648, 936.1837158
1: -214.4616547, 865.6424561, -219.7311096, 888.0158081, -1102.4772949, 1085.3735352
2: -218.7769775, 857.2067871, -223.9974518, 879.2314453, -1098.0083008, 1081.2042236
3: -346.8608398, 899.7705688, -355.7673950, 922.9997559, -1269.8605957, 1255.5378418
4: -349.8637695, 857.6834106, -358.6697693, 879.5055542, -1229.3692627, 1216.3531494

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2809122, upper bound: 853.2643276
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2793620, upper bound: 853.2634073
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -157.9430389, 703.3171387, -868.6638794, 894.8601685
1: -207.0206451, 838.7129517, -197.7702942, 800.6807251, -1007.7013550, 1036.4832764
2: -211.4122925, 830.5660400, -202.2394409, 793.0823975, -1004.4946899, 1032.8054199
3: -335.2691040, 871.3525391, -319.9077759, 831.7758789, -1167.0446777, 1191.2602539
4: -338.4065247, 830.8461914, -323.1899719, 793.5729980, -1131.9794922, 1154.0361328

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009349, upper bound: 853.3001474
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -165.3467407, 736.9172363, -902.2639771, 902.2639771
1: -207.0206451, 838.7129517, -207.0206451, 838.7129517, -1045.7336426, 1045.7336426
2: -211.4122925, 830.5660400, -211.4122925, 830.5660400, -1041.9782715, 1041.9782715
3: -335.2691040, 871.3525391, -335.2691040, 871.3525391, -1206.6215820, 1206.6215820
4: -338.4065247, 830.8461914, -338.4065247, 830.8461914, -1169.2526855, 1169.2526855

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009349, upper bound: 853.3001474
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -157.9430389, 703.3171387, -879.0449829, 938.1967773
1: -219.7311096, 888.0158081, -197.7702942, 800.6807251, -1020.4118042, 1085.7858887
2: -223.9974518, 879.2314453, -202.2394409, 793.0823975, -1017.0798340, 1081.4708252
3: -355.7673950, 922.9997559, -319.9077759, 831.7758789, -1187.5429688, 1242.9074707
4: -358.6697693, 879.5055542, -323.1899719, 793.5729980, -1152.2426758, 1202.6955566

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005973, upper bound: 853.3009798
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015355, upper bound: 853.3019912
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2630346, upper bound: 853.2798728
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -165.3467407, 736.9172363, -912.6450806, 945.6005859
1: -219.7311096, 888.0158081, -207.0206451, 838.7129517, -1058.4440918, 1095.0364990
2: -223.9974518, 879.2314453, -211.4122925, 830.5660400, -1054.5634766, 1090.6437988
3: -355.7673950, 922.9997559, -335.2691040, 871.3525391, -1227.1198730, 1258.2687988
4: -358.6697693, 879.5055542, -338.4065247, 830.8461914, -1189.5158691, 1217.9121094

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005973, upper bound: 853.3009798
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015355, upper bound: 853.3019912
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2630346, upper bound: 853.2798728
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -171.4976959, 760.4559326, -925.8026123, 908.4148560
1: -207.0206451, 838.7129517, -214.4616547, 865.6424561, -1072.6630859, 1053.1745605
2: -211.4122925, 830.5660400, -218.7769775, 857.2067871, -1068.6191406, 1049.3428955
3: -335.2691040, 871.3525391, -346.8608398, 899.7705688, -1235.0395508, 1218.2133789
4: -338.4065247, 830.8461914, -349.8637695, 857.6834106, -1196.0899658, 1180.7098389

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979147, upper bound: 853.2959847
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -165.3467407, 736.9172363, -175.7278595, 780.2539062, -945.6005859, 912.6450806
1: -207.0206451, 838.7129517, -219.7311096, 888.0158081, -1095.0364990, 1058.4440918
2: -211.4122925, 830.5660400, -223.9974518, 879.2314453, -1090.6437988, 1054.5634766
3: -335.2691040, 871.3525391, -355.7673950, 922.9997559, -1258.2687988, 1227.1198730
4: -338.4065247, 830.8461914, -358.6697693, 879.5055542, -1217.9121094, 1189.5159912

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979147, upper bound: 853.2959847
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -171.4976959, 760.4559326, -936.1837769, 951.7514648
1: -219.7311096, 888.0158081, -214.4616547, 865.6424561, -1085.3735352, 1102.4772949
2: -223.9974518, 879.2314453, -218.7769775, 857.2067871, -1081.2042236, 1098.0083008
3: -355.7673950, 922.9997559, -346.8608398, 899.7705688, -1255.5378418, 1269.8605957
4: -358.6697693, 879.5055542, -349.8637695, 857.6834106, -1216.3531494, 1229.3692627

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2812207, upper bound: 853.2639257
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -175.7278595, 780.2539062, -175.7278595, 780.2539062, -955.9816895, 955.9816895
1: -219.7311096, 888.0158081, -219.7311096, 888.0158081, -1107.7467041, 1107.7467041
2: -223.9974518, 879.2314453, -223.9974518, 879.2314453, -1103.2288818, 1103.2288818
3: -355.7673950, 922.9997559, -355.7673950, 922.9997559, -1278.7670898, 1278.7670898
4: -358.6697693, 879.5055542, -358.6697693, 879.5055542, -1238.1752930, 1238.1752930

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2812207, upper bound: 853.2639257
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.16 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3014599, upper bound: 853.3002908
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2995889
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3014599, upper bound: 853.3002908
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2995889
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3023592, upper bound: 853.3016342
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3023592, upper bound: 853.3016342
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2979258, upper bound: 853.2962687
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2979258, upper bound: 853.2962687
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2809122, upper bound: 853.2643276
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2793620, upper bound: 853.2634073
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2809122, upper bound: 853.2643276
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2793620, upper bound: 853.2634073
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3009349, upper bound: 853.3001474
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3009349, upper bound: 853.3001474
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3015355, upper bound: 853.3019912
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2630346, upper bound: 853.2798728
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.3015355, upper bound: 853.3019912
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2630346, upper bound: 853.2798728
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2979147, upper bound: 853.2959847
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2979147, upper bound: 853.2959847
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2812207, upper bound: 853.2639257
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2812207, upper bound: 853.2639257
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.16
Output dim: 0, lower bound: -853.2554213, upper bound: 853.2554213

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -156.7290649, 697.9744873, -835.8710938, 773.0100708
1: -172.6246338, 701.4801025, -196.2534027, 794.6039429, -967.2285767, 897.7335205
2: -176.5765839, 694.5836182, -200.6870880, 787.0542603, -963.6308594, 895.2706909
3: -279.6128540, 728.3481445, -317.4566650, 825.4671631, -1105.0799561, 1045.8045654
4: -282.3982544, 694.6760254, -320.7378235, 787.5434570, -1069.9416504, 1015.4138184

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011437, upper bound: 853.3018811
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -157.7457123, 702.4475098, -857.3630981, 847.7192993
1: -193.9920044, 785.5104980, -197.5241241, 799.6919556, -993.6839600, 983.0346069
2: -198.3775940, 778.0156860, -201.9879913, 792.1009521, -990.4785156, 980.0036621
3: -313.7880249, 816.0270386, -319.5089111, 830.7497559, -1144.5378418, 1135.5358887
4: -317.0524902, 778.5471191, -322.7902832, 792.5944214, -1109.6468506, 1101.3374023

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -170.0299988, 754.0557251, -891.9523926, 786.3109741
1: -172.6246338, 701.4801025, -212.6285706, 858.3634644, -1030.9880371, 914.1085205
2: -176.5765839, 694.5836182, -216.9078827, 849.9818115, -1026.5583496, 911.4915161
3: -279.6128540, 728.3481445, -343.9075012, 892.2069702, -1171.8198242, 1072.2556152
4: -282.3982544, 694.6760254, -346.9234924, 850.4450073, -1132.8431396, 1041.5994873

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -171.2458801, 759.3678589, -914.2834473, 861.2194824
1: -193.9920044, 785.5104980, -214.1477203, 864.4054565, -1058.3973389, 999.6582031
2: -198.3775940, 778.0156860, -218.4577942, 855.9780273, -1054.3553467, 996.4735107
3: -313.7880249, 816.0270386, -346.3555603, 898.4831543, -1212.2709961, 1162.3823242
4: -317.0524902, 778.5471191, -349.3580933, 856.4542236, -1173.5065918, 1127.9052734

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002049, upper bound: 853.2992671
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2995889
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -157.9430389, 703.3171387, -870.0597534, 898.1791382
1: -208.5561523, 842.6534424, -197.7702942, 800.6807251, -1009.2368164, 1040.4235840
2: -212.7993317, 834.3977051, -202.2394409, 793.0823975, -1005.8817139, 1036.6372070
3: -337.4005737, 875.8064575, -319.9077759, 831.7758789, -1169.1763916, 1195.7141113
4: -340.4402466, 834.8568115, -323.1899719, 793.5729980, -1134.0131836, 1158.0467529

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997989, upper bound: 853.3011631
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -157.6940460, 702.3458862, -876.5637817, 931.8759766
1: -217.9788666, 881.2302246, -197.4634094, 799.5730591, -1017.5518188, 1078.6936035
2: -222.4038544, 872.6578369, -201.9283905, 791.9830933, -1014.3869629, 1074.5861816
3: -352.7824707, 915.8316650, -319.4345093, 830.6164551, -1183.3989258, 1235.2655029
4: -355.6772156, 873.3172607, -322.7106323, 792.4718018, -1148.1489258, 1196.0278320

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001209, upper bound: 853.3013934
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -171.4976959, 760.4559326, -927.1985474, 911.7338257
1: -208.5561523, 842.6534424, -214.4616547, 865.6424561, -1074.1986084, 1057.1151123
2: -212.7993317, 834.3977051, -218.7769775, 857.2067871, -1070.0061035, 1053.1744385
3: -337.4005737, 875.8064575, -346.8608398, 899.7705688, -1237.1711426, 1222.6672363
4: -340.4402466, 834.8568115, -349.8637695, 857.6834106, -1198.1236572, 1184.7203369

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -171.2814484, 759.6338501, -933.8517456, 945.4632568
1: -217.9788666, 881.2302246, -214.1972656, 864.7041016, -1082.6828613, 1095.4274902
2: -222.4038544, 872.6578369, -218.5093231, 856.2763672, -1078.6801758, 1091.1671143
3: -352.7824707, 915.8316650, -346.4570007, 898.7871094, -1251.5695801, 1262.2885742
4: -355.6772156, 873.3172607, -349.4494934, 856.7528687, -1212.4300537, 1222.7667236

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007242, upper bound: 853.3006109
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009987, upper bound: 853.3009987
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -164.0718536, 731.3773804, -869.2739868, 780.3529663
1: -172.6246338, 701.4801025, -205.4293518, 832.4129028, -1005.0375366, 906.9094238
2: -176.5765839, 694.5836182, -209.7862091, 824.3142090, -1000.8908081, 904.3698120
3: -279.6128540, 728.3481445, -332.7063599, 864.8073730, -1144.4201660, 1061.0544434
4: -282.3982544, 694.6760254, -335.8448486, 824.5880737, -1106.9860840, 1030.5208740

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007321, upper bound: 853.3001469
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3002044
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -165.1505585, 736.0548096, -890.9703979, 855.1241455
1: -193.9920044, 785.5104980, -206.7757416, 837.7327271, -1031.7247314, 992.2862549
2: -198.3775940, 778.0156860, -211.1625519, 829.5923462, -1027.9697266, 989.1782227
3: -313.7880249, 816.0270386, -334.8721619, 870.3347168, -1184.1228027, 1150.8991699
4: -317.0524902, 778.5471191, -338.0094604, 829.8740234, -1146.9262695, 1116.5566406

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -174.3090210, 774.0638428, -911.9605103, 790.5899658
1: -172.6246338, 701.4801025, -217.9582520, 880.9744263, -1053.5988770, 919.4382935
2: -176.5765839, 694.5836182, -222.1931000, 872.2431641, -1048.8195801, 916.7767334
3: -279.6128540, 728.3481445, -352.9091187, 915.6820068, -1195.2949219, 1081.2570801
4: -282.3982544, 694.6760254, -355.8264771, 872.5070801, -1154.9049072, 1050.5024414

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976906, upper bound: 853.2960462
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979258, upper bound: 853.2961569
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -175.4736176, 779.1558838, -934.0714722, 865.4472656
1: -193.9920044, 785.5104980, -219.4139404, 886.7673950, -1080.7593994, 1004.9244385
2: -198.3775940, 778.0156860, -223.6753082, 877.9906006, -1076.3680420, 1001.6909790
3: -313.7880249, 816.0270386, -355.2570801, 921.7007446, -1235.4887695, 1171.2840576
4: -317.0524902, 778.5471191, -358.1604004, 878.2644653, -1195.3166504, 1136.7075195

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2966335, upper bound: 853.2954234
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2969516, upper bound: 853.2956327
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2798728, upper bound: 853.2633247
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -156.7290649, 697.9744873, -842.7813721, 803.6826172
1: -181.2450409, 736.1990356, -196.2534027, 794.6039429, -975.8489990, 932.4524536
2: -185.1202850, 728.8364258, -200.6870880, 787.0542603, -972.1744385, 929.5234985
3: -293.8229980, 764.5282593, -317.4566650, 825.4671631, -1119.2899170, 1081.9847412
4: -296.4824219, 728.7455444, -320.7378235, 787.5434570, -1084.0258789, 1049.4833984

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004739, upper bound: 853.3008962
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -157.7457123, 702.4475098, -864.5451050, 880.3908081
1: -202.9634247, 822.4923096, -197.5241241, 799.6919556, -1002.6553345, 1020.0164185
2: -207.2748718, 814.4534302, -201.9879913, 792.1009521, -999.3758545, 1016.4414062
3: -328.6972656, 854.5055542, -319.5089111, 830.7497559, -1159.4470215, 1174.0144043
4: -331.8298645, 814.7550659, -322.7902832, 792.5944214, -1124.4241943, 1137.5451660

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -164.0718536, 731.3773804, -876.1842651, 811.0254517
1: -181.2450409, 736.1990356, -205.4293518, 832.4129028, -1013.6579590, 941.6282959
2: -185.1202850, 728.8364258, -209.7862091, 824.3142090, -1009.4343872, 938.6226196
3: -293.8229980, 764.5282593, -332.7063599, 864.8073730, -1158.6303711, 1097.2346191
4: -296.4824219, 728.7455444, -335.8448486, 824.5880737, -1121.0703125, 1064.5903320

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004726, upper bound: 853.2994984
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -165.1505585, 736.0548096, -898.1523438, 887.7957153
1: -202.9634247, 822.4923096, -206.7757416, 837.7327271, -1040.6961670, 1029.2680664
2: -207.2748718, 814.4534302, -211.1625519, 829.5923462, -1036.8671875, 1025.6159668
3: -328.6972656, 854.5055542, -334.8721619, 870.3347168, -1199.0318604, 1189.3776855
4: -331.8298645, 814.7550659, -338.0094604, 829.8740234, -1161.7036133, 1152.7645264

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -170.9987030, 760.2313232, -157.9430389, 703.3171387, -874.3157959, 918.1742554
1: -213.8578033, 865.2389526, -197.7702942, 800.6807251, -1014.5384521, 1063.0092773
2: -218.0611115, 856.6461182, -202.2394409, 793.0823975, -1011.1434937, 1058.8854980
3: -346.3732300, 899.2537842, -319.9077759, 831.7758789, -1178.1489258, 1219.1612549
4: -349.3176575, 856.9051514, -323.1899719, 793.5729980, -1142.8906250, 1180.0950928

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960462, upper bound: 853.2976906
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -170.9987030, 760.2313232, -165.3467407, 736.9172363, -907.9158325, 925.5780640
1: -213.8578033, 865.2389526, -207.0206451, 838.7129517, -1052.5708008, 1072.2596436
2: -218.0611115, 856.6461182, -211.4122925, 830.5660400, -1048.6270752, 1068.0583496
3: -346.3732300, 899.2537842, -335.2691040, 871.3525391, -1217.7258301, 1234.5225830
4: -349.3176575, 856.9051514, -338.4065247, 830.8461914, -1180.1638184, 1195.3116455

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2953480, upper bound: 853.2973471
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -170.0299988, 754.0557251, -898.8626099, 816.9834595
1: -181.2450409, 736.1990356, -212.6285706, 858.3634644, -1039.6082764, 948.8273926
2: -185.1202850, 728.8364258, -216.9078827, 849.9818115, -1035.1020508, 945.7443237
3: -293.8229980, 764.5282593, -343.9075012, 892.2069702, -1186.0300293, 1108.4356689
4: -296.4824219, 728.7455444, -346.9234924, 850.4450073, -1146.9273682, 1075.6690674

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987409, upper bound: 853.2968095
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2939959, upper bound: 853.2896504
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -171.2458801, 759.3678589, -921.4653320, 893.8910522
1: -202.9634247, 822.4923096, -214.1477203, 864.4054565, -1067.3687744, 1036.6400146
2: -207.2748718, 814.4534302, -218.4577942, 855.9780273, -1063.2528076, 1032.9112549
3: -328.6972656, 854.5055542, -346.3555603, 898.4831543, -1227.1799316, 1200.8610840
4: -331.8298645, 814.7550659, -349.3580933, 856.4542236, -1188.2839355, 1164.1131592

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2962966, upper bound: 853.2956183
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2933772, upper bound: 853.2900110
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -174.3090210, 774.0638428, -918.8707275, 821.2624512
1: -181.2450409, 736.1990356, -217.9582520, 880.9744263, -1062.2192383, 954.1571045
2: -185.1202850, 728.8364258, -222.1931000, 872.2431641, -1057.3631592, 951.0295410
3: -293.8229980, 764.5282593, -352.9091187, 915.6820068, -1209.5050049, 1117.4371338
4: -296.4824219, 728.7455444, -355.8264771, 872.5070801, -1168.9892578, 1084.5720215

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2973471, upper bound: 853.2953480
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979147, upper bound: 853.2958548
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -175.4736176, 779.1558838, -941.2534790, 898.1188354
1: -202.9634247, 822.4923096, -219.4139404, 886.7673950, -1089.7308350, 1041.9062500
2: -207.2748718, 814.4534302, -223.6753082, 877.9906006, -1085.2655029, 1038.1287842
3: -328.6972656, 854.5055542, -355.2570801, 921.7007446, -1250.3978271, 1209.7626953
4: -331.8298645, 814.7550659, -358.1604004, 878.2644653, -1210.0941162, 1172.9152832

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960073, upper bound: 853.2946302
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2798754, upper bound: 853.2630393
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.35 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3011437, upper bound: 853.3018811
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3002049, upper bound: 853.2992671
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3004331, upper bound: 853.2995889
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2997989, upper bound: 853.3011631
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3001209, upper bound: 853.3013934
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3010605, upper bound: 853.3010605
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3007242, upper bound: 853.3006109
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3009987, upper bound: 853.3009987
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3002044
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2969558, upper bound: 853.2956327
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2969516, upper bound: 853.2956327
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2798728, upper bound: 853.2633247
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3004739, upper bound: 853.3008962
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2960462, upper bound: 853.2976906
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2953480, upper bound: 853.2973471
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2987409, upper bound: 853.2968095
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2939959, upper bound: 853.2896504
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2962966, upper bound: 853.2956183
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2933772, upper bound: 853.2900110
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2960139, upper bound: 853.2946302
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2960073, upper bound: 853.2946302
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.35
Output dim: 0, lower bound: -853.2798754, upper bound: 853.2630393

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -153.2341461, 682.8717651, -820.7683716, 769.5151978
1: -172.6246338, 701.4801025, -191.8938904, 777.4130249, -950.0375977, 893.3739624
2: -176.5765839, 694.5836182, -196.2630310, 770.0264282, -946.6030273, 890.8466797
3: -279.6128540, 728.3481445, -310.4435730, 807.5858765, -1087.1987305, 1038.7917480
4: -282.3982544, 694.6760254, -313.7052002, 770.5253906, -1052.9234619, 1008.3811646

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -137.6880188, 615.4636841, -159.7838593, 712.9316406, -850.6195679, 775.2475586
1: -172.3667603, 700.5476685, -200.2044525, 811.5670166, -983.9337769, 900.7521362
2: -176.3152771, 693.6602783, -204.7386475, 803.9956665, -980.3108521, 898.3989258
3: -279.2152405, 727.3725586, -324.0181885, 842.9988403, -1122.2139893, 1051.3907471
4: -281.9947510, 693.7512207, -327.1376343, 804.7034302, -1086.6981201, 1020.8888550

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016558, upper bound: 853.3013192
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021848, upper bound: 853.3017866
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -137.8967133, 616.2811279, -771.1967163, 827.8703003
1: -193.9920044, 785.5104980, -172.6246338, 701.4801025, -895.4721069, 958.1351318
2: -198.3775940, 778.0156860, -176.5765839, 694.5836182, -892.9611816, 954.5922852
3: -313.7880249, 816.0270386, -279.6128540, 728.3481445, -1042.1362305, 1095.6397705
4: -317.0524902, 778.5471191, -282.3982544, 694.6760254, -1011.7285156, 1060.9450684

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3001867
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -154.9156036, 689.9736938, -844.8892822, 844.8892822
1: -193.9920044, 785.5104980, -193.9920044, 785.5104980, -979.5025024, 979.5025024
2: -198.3775940, 778.0156860, -198.3775940, 778.0156860, -976.3933105, 976.3933105
3: -313.7880249, 816.0270386, -313.7880249, 816.0270386, -1129.8150635, 1129.8150635
4: -317.0524902, 778.5471191, -317.0524902, 778.5471191, -1095.5994873, 1095.5993652

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3001867
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3013621
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -165.2764587, 733.8463745, -871.7429810, 781.5574951
1: -172.6246338, 701.4801025, -206.7252960, 835.3848877, -1008.0095215, 908.2053833
2: -176.5765839, 694.5836182, -210.9334106, 827.1862183, -1003.7627563, 905.5170288
3: -279.6128540, 728.3481445, -334.4519958, 868.2522583, -1147.8649902, 1062.8001709
4: -282.3982544, 694.6760254, -337.5020447, 827.6334229, -1110.0314941, 1032.1781006

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -137.6880188, 615.4636841, -172.8704681, 768.2937012, -905.9816895, 788.3341064
1: -172.3667603, 700.5476685, -216.2959595, 874.5336914, -1046.9003906, 916.8436279
2: -176.3152771, 693.6602783, -220.6899109, 866.0119629, -1042.3272705, 914.3501587
3: -279.2152405, 727.3725586, -350.0679932, 908.8720703, -1188.0872803, 1077.4405518
4: -281.9947510, 693.7512207, -352.9742737, 866.6624146, -1148.6571045, 1046.7254639

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -166.4912109, 739.1508179, -894.0664062, 856.4648438
1: -193.9920044, 785.5104980, -208.2427063, 841.4191284, -1035.4111328, 993.7531128
2: -198.3775940, 778.0156860, -212.4807587, 833.1720581, -1031.5495605, 990.4964600
3: -313.7880249, 816.0270386, -336.8959045, 874.5217285, -1188.3098145, 1152.9228516
4: -317.0524902, 778.5471191, -339.9353027, 833.6307983, -1150.6829834, 1118.4824219

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981616, upper bound: 853.2981366
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2981616, upper bound: 853.2992671
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -154.6401672, 688.8912354, -173.9344177, 772.9648438, -927.6049805, 862.8255615
1: -193.6525879, 784.2756348, -217.6252747, 879.8460083, -1073.4985352, 1001.9008789
2: -198.0339813, 776.7918091, -222.0443573, 871.2821655, -1069.3161621, 998.8361816
3: -313.2620239, 814.7352295, -352.2157898, 914.3903809, -1227.6523438, 1166.9510498
4: -316.5205688, 777.3224487, -355.1098938, 871.9403687, -1188.4609375, 1132.4323730

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983844, upper bound: 853.2984253
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983844, upper bound: 853.2995889
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -165.2764587, 733.8463745, -137.8967133, 616.2811279, -781.5574951, 871.7429810
1: -206.7252960, 835.3848877, -172.6246338, 701.4801025, -908.2053833, 1008.0095215
2: -210.9334106, 827.1862183, -176.5765839, 694.5836182, -905.5170288, 1003.7628174
3: -334.4519958, 868.2522583, -279.6128540, 728.3481445, -1062.8001709, 1147.8649902
4: -337.5020447, 827.6334229, -282.3982544, 694.6760254, -1032.1781006, 1110.0314941

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2990681, upper bound: 853.3008199
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997017, upper bound: 853.3011318
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -166.4912109, 739.1508179, -154.9156036, 689.9736938, -856.4649048, 894.0664062
1: -208.2427063, 841.4191284, -193.9920044, 785.5104980, -993.7531128, 1035.4111328
2: -212.4807587, 833.1720581, -198.3775940, 778.0156860, -990.4964600, 1031.5495605
3: -336.8959045, 874.5217285, -313.7880249, 816.0270386, -1152.9228516, 1188.3098145
4: -339.9353027, 833.6307983, -317.0524902, 778.5471191, -1118.4824219, 1150.6829834

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2986461, upper bound: 853.3001666
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991123, upper bound: 853.3001748
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -172.8704681, 768.2937012, -137.6880188, 615.4636841, -788.3341064, 905.9816895
1: -216.2959595, 874.5336914, -172.3667603, 700.5476685, -916.8436279, 1046.9003906
2: -220.6899109, 866.0119629, -176.3152771, 693.6602783, -914.3501587, 1042.3272705
3: -350.0679932, 908.8720703, -279.2152405, 727.3725586, -1077.4405518, 1188.0872803
4: -352.9742737, 866.6624146, -281.9947510, 693.7512207, -1046.7254639, 1148.6571045

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993704, upper bound: 853.3010442
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000363, upper bound: 853.3013525
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3004331
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -173.9344177, 772.9648438, -154.6401672, 688.8912354, -862.8255615, 927.6049805
1: -217.6252747, 879.8460083, -193.6525879, 784.2756348, -1001.9008789, 1073.4985352
2: -222.0443573, 871.2821655, -198.0339813, 776.7918091, -998.8361816, 1069.3161621
3: -352.2157898, 914.3903809, -313.2620239, 814.7352295, -1166.9510498, 1227.6523438
4: -355.1098938, 871.9403687, -316.5205688, 777.3224487, -1132.4323730, 1188.4609375

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2986461, upper bound: 853.3003897
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994381, upper bound: 853.3003921
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -166.7426147, 740.2362061, -906.9788208, 906.9788208
1: -208.5561523, 842.6534424, -208.5561523, 842.6534424, -1051.2095947, 1051.2095947
2: -212.7993317, 834.3977051, -212.7993317, 834.3977051, -1047.1970215, 1047.1970215
3: -337.4005737, 875.8064575, -337.4005737, 875.8064575, -1213.2070312, 1213.2070312
4: -340.4402466, 834.8568115, -340.4402466, 834.8568115, -1175.2969971, 1175.2969971

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3008786
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3015926
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -166.7426147, 740.2362061, -174.2179718, 774.1820068, -940.9246216, 914.4541016
1: -208.5561523, 842.6534424, -217.9788666, 881.2302246, -1089.7863770, 1060.6322021
2: -212.7993317, 834.3977051, -222.4038544, 872.6578369, -1085.4571533, 1056.8015137
3: -337.4005737, 875.8064575, -352.7824707, 915.8316650, -1253.2320557, 1228.5888672
4: -340.4402466, 834.8568115, -355.6772156, 873.3172607, -1213.7575684, 1190.5340576

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3008786
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023444, upper bound: 853.3015926
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -174.2179718, 774.1820068, -165.8609924, 735.4425049, -909.6604004, 940.0429077
1: -217.9788666, 881.2302246, -207.3368073, 837.2872925, -1055.2658691, 1088.5670166
2: -222.4038544, 872.6578369, -211.5244293, 828.9431763, -1051.3470459, 1084.1820068
3: -352.7824707, 915.8316650, -335.4260864, 870.2937622, -1223.0761719, 1251.2575684
4: -355.6772156, 873.3172607, -338.5411682, 829.2362671, -1184.9134521, 1211.8583984

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003339, upper bound: 853.3003339
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003339, upper bound: 853.3006109
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -173.1892853, 769.6303711, -166.8730469, 741.5205078, -914.7097778, 936.5034180
1: -216.6891937, 876.0611572, -208.7063904, 844.0096436, -1060.6988525, 1084.7675781
2: -221.0960083, 867.5154419, -212.9660187, 835.7252197, -1056.8210449, 1080.4814453
3: -350.6853943, 910.4580688, -337.7717590, 877.1044922, -1227.7895508, 1248.2298584
4: -353.5993347, 868.1645508, -340.6423950, 836.1918945, -1189.7912598, 1208.8068848

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006109, upper bound: 853.3007242
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006109, upper bound: 853.3009987
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -160.6674652, 716.8948364, -854.7914429, 776.9484863
1: -172.6246338, 701.4801025, -201.1900024, 815.9338989, -988.5585327, 902.6700439
2: -176.5765839, 694.5836182, -205.4857178, 807.9945679, -984.5711670, 900.0693359
3: -279.6128540, 728.3481445, -325.9268494, 847.6497192, -1127.2625732, 1054.2749023
4: -282.3982544, 694.6760254, -329.0484009, 808.2494507, -1090.6473389, 1023.7244263

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -137.6880188, 615.4636841, -166.5481262, 743.6938477, -881.3818359, 782.0117798
1: -172.3667603, 700.5476685, -208.6714325, 846.3873901, -1018.7541504, 909.2190552
2: -176.3152771, 693.6602783, -213.1143951, 838.2883911, -1014.6035767, 906.7745972
3: -279.2152405, 727.3725586, -338.0823364, 879.2489624, -1158.4642334, 1065.4548340
4: -281.9947510, 693.7512207, -341.0727234, 838.7627563, -1120.7573242, 1034.8239746

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3001950
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3002044
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -144.8068848, 646.9536133, -801.8692017, 834.7805786
1: -193.9920044, 785.5104980, -181.2450409, 736.1990356, -930.1910400, 966.7555542
2: -198.3775940, 778.0156860, -185.1202850, 728.8364258, -927.2139893, 963.1359863
3: -313.7880249, 816.0270386, -293.8229980, 764.5282593, -1078.3162842, 1109.8498535
4: -317.0524902, 778.5471191, -296.4824219, 728.7455444, -1045.7978516, 1075.0294189

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2983370
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2998001
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -162.0976410, 722.6452026, -877.5607910, 852.0712280
1: -193.9920044, 785.5104980, -202.9634247, 822.4923096, -1016.4843140, 988.4739380
2: -198.3775940, 778.0156860, -207.2748718, 814.4534302, -1012.8309326, 985.2905273
3: -313.7880249, 816.0270386, -328.6972656, 854.5055542, -1168.2935791, 1144.7241211
4: -317.0524902, 778.5471191, -331.8298645, 814.7550659, -1131.8071289, 1110.3768311

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2983370
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -157.8990173, 700.5250244, -838.4216309, 774.1801147
1: -172.6246338, 701.4801025, -197.3341217, 797.1621094, -969.7867432, 898.8141479
2: -176.5765839, 694.5836182, -201.0137177, 789.2255859, -965.8021851, 895.5973511
3: -279.6128540, 728.3481445, -319.5162048, 828.4740601, -1108.0869141, 1047.8642578
4: -282.3982544, 694.6760254, -321.8711243, 789.2375488, -1071.6354980, 1016.5471191

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2941821, upper bound: 853.2936496
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2978905, upper bound: 853.2959271
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -137.8967133, 616.2811279, -171.5457764, 762.1805420, -900.0770874, 787.8268433
1: -172.6246338, 701.4801025, -214.5125122, 867.4678955, -1040.0924072, 915.9926147
2: -176.5765839, 694.5836182, -218.6978760, 858.8092041, -1035.3856201, 913.2814331
3: -279.6128540, 728.3481445, -347.3695679, 901.6154175, -1181.2281494, 1075.7174072
4: -282.3982544, 694.6760254, -350.2860107, 859.0719604, -1141.4699707, 1044.9620361

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2941821, upper bound: 853.2936496
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2978905, upper bound: 853.2959271
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -154.9156036, 689.9736938, -170.7339478, 759.0927734, -914.0083618, 860.7075195
1: -193.9920044, 785.5104980, -213.5276031, 863.9444580, -1057.9365234, 999.0380859
2: -198.3775940, 778.0156860, -217.7257538, 855.3594360, -1053.7369385, 995.7414551
3: -313.7880249, 816.0270386, -345.8427429, 897.9061890, -1211.6942139, 1161.8697510
4: -317.0524902, 778.5471191, -348.7878418, 855.6179810, -1172.6701660, 1127.3349609

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2798728, upper bound: 853.2633247
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2798728, upper bound: 853.2633247
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -153.2341461, 682.8717651, -827.6785889, 800.1876831
1: -181.2450409, 736.1990356, -191.8938904, 777.4130249, -958.6580811, 928.0928345
2: -185.1202850, 728.8364258, -196.2630310, 770.0264282, -955.1467285, 925.0994873
3: -293.8229980, 764.5282593, -310.4435730, 807.5858765, -1101.4088135, 1074.9718018
4: -296.4824219, 728.7455444, -313.7052002, 770.5253906, -1067.0078125, 1042.4506836

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -144.5762024, 646.0222778, -159.7838593, 712.9316406, -857.5076904, 805.8061523
1: -180.9601135, 735.1380615, -200.2044525, 811.5670166, -992.5270996, 935.3424683
2: -184.8313751, 727.7849121, -204.7386475, 803.9956665, -988.8270264, 932.5235596
3: -293.3773499, 763.4212036, -324.0181885, 842.9988403, -1136.3762207, 1087.4392090
4: -296.0311584, 727.6936646, -327.1376343, 804.7034302, -1100.7346191, 1054.8311768

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016675, upper bound: 853.3008962
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014921, upper bound: 853.3007124
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -137.8967133, 616.2811279, -778.3786011, 860.5418091
1: -202.9634247, 822.4923096, -172.6246338, 701.4801025, -904.4434814, 995.1169434
2: -207.2748718, 814.4534302, -176.5765839, 694.5836182, -901.8585205, 991.0299683
3: -328.6972656, 854.5055542, -279.6128540, 728.3481445, -1057.0451660, 1134.1184082
4: -331.8298645, 814.7550659, -282.3982544, 694.6760254, -1026.5057373, 1097.1529541

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998001, upper bound: 853.2999978
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985446, upper bound: 853.2991498
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -154.9156036, 689.9736938, -852.0712280, 877.5607910
1: -202.9634247, 822.4923096, -193.9920044, 785.5104980, -988.4739380, 1016.4843140
2: -207.2748718, 814.4534302, -198.3775940, 778.0156860, -985.2905273, 1012.8309326
3: -328.6972656, 854.5055542, -313.7880249, 816.0270386, -1144.7239990, 1168.2935791
4: -331.8298645, 814.7550659, -317.0524902, 778.5471191, -1110.3768311, 1131.8071289

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998001, upper bound: 853.2999978
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985446, upper bound: 853.2991498
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -160.6674652, 716.8948364, -861.7017212, 807.6209717
1: -181.2450409, 736.1990356, -201.1900024, 815.9338989, -997.1789551, 937.3889771
2: -185.1202850, 728.8364258, -205.4857178, 807.9945679, -993.1148071, 934.3221436
3: -293.8229980, 764.5282593, -325.9268494, 847.6497192, -1141.4726562, 1090.4550781
4: -296.4824219, 728.7455444, -329.0484009, 808.2494507, -1104.7316895, 1057.7938232

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -144.5762024, 646.0222778, -166.5481262, 743.6938477, -888.2699585, 812.5704346
1: -180.9601135, 735.1380615, -208.6714325, 846.3873901, -1027.3474121, 943.8093872
2: -184.8313751, 727.7849121, -213.1143951, 838.2883911, -1023.1197510, 940.8992310
3: -293.3773499, 763.4212036, -338.0823364, 879.2489624, -1172.6263428, 1101.5034180
4: -296.0311584, 727.6936646, -341.0727234, 838.7627563, -1134.7937012, 1068.7662354

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -144.8068848, 646.9536133, -809.0511475, 867.4520874
1: -202.9634247, 822.4923096, -181.2450409, 736.1990356, -939.1623535, 1003.7373657
2: -207.2748718, 814.4534302, -185.1202850, 728.8364258, -936.1113281, 999.5736084
3: -328.6972656, 854.5055542, -293.8229980, 764.5282593, -1093.2253418, 1148.3286133
4: -331.8298645, 814.7550659, -296.4824219, 728.7455444, -1060.5753174, 1111.2371826

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974063, upper bound: 853.2978749
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -162.0976410, 722.6452026, -884.7427368, 884.7427368
1: -202.9634247, 822.4923096, -202.9634247, 822.4923096, -1025.4555664, 1025.4555664
2: -207.2748718, 814.4534302, -207.2748718, 814.4534302, -1021.7282715, 1021.7282715
3: -328.6972656, 854.5055542, -328.6972656, 854.5055542, -1183.2028809, 1183.2028809
4: -331.8298645, 814.7550659, -331.8298645, 814.7550659, -1146.5844727, 1146.5844727

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2974063, upper bound: 853.2978749
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -169.5963440, 754.1082153, -137.8967133, 616.2811279, -785.8774414, 892.0048828
1: -212.1077728, 858.2727051, -172.6246338, 701.4801025, -913.5878296, 1030.8973389
2: -216.2788696, 849.7365112, -176.5765839, 694.5836182, -910.8624878, 1026.3131104
3: -343.5503540, 892.0159912, -279.6128540, 728.3481445, -1071.8984375, 1171.6285400
4: -346.5053101, 849.9884033, -282.3982544, 694.6760254, -1041.1813965, 1132.3864746

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960049, upper bound: 853.2976906
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -170.7339478, 759.0927734, -154.9156036, 689.9736938, -860.7075806, 914.0083618
1: -213.5276031, 863.9444580, -193.9920044, 785.5104980, -999.0380859, 1057.9364014
2: -217.7257538, 855.3594360, -198.3775940, 778.0156860, -995.7414551, 1053.7369385
3: -345.8427429, 897.9061890, -313.7880249, 816.0270386, -1161.8697510, 1211.6942139
4: -348.7878418, 855.6179810, -317.0524902, 778.5471191, -1127.3348389, 1172.6701660

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2954234, upper bound: 853.2966305
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955900, upper bound: 853.2968361
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -169.5963440, 754.1082153, -144.8068848, 646.9536133, -816.5499268, 898.9151001
1: -212.1077728, 858.2727051, -181.2450409, 736.1990356, -948.3065796, 1039.5177002
2: -216.2788696, 849.7365112, -185.1202850, 728.8364258, -945.1152954, 1034.8568115
3: -343.5503540, 892.0159912, -293.8229980, 764.5282593, -1108.0786133, 1185.8385010
4: -346.5053101, 849.9884033, -296.4824219, 728.7455444, -1075.2508545, 1146.4708252

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -170.7339478, 759.0927734, -162.0976410, 722.6452026, -893.3790894, 921.1902466
1: -213.5276031, 863.9444580, -202.9634247, 822.4923096, -1036.0198975, 1066.9078369
2: -217.7257538, 855.3594360, -207.2748718, 814.4534302, -1032.1791992, 1062.6342773
3: -345.8427429, 897.9061890, -328.6972656, 854.5055542, -1200.3482666, 1226.6031494
4: -348.7878418, 855.6179810, -331.8298645, 814.7550659, -1163.5426025, 1187.4475098

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
time: 1.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -165.2764587, 733.8463745, -878.6531982, 812.2299805
1: -181.2450409, 736.1990356, -206.7252960, 835.3848877, -1016.6299438, 942.9242554
2: -185.1202850, 728.8364258, -210.9334106, 827.1862183, -1012.3063354, 939.7698364
3: -293.8229980, 764.5282593, -334.4519958, 868.2522583, -1162.0750732, 1098.9802246
4: -296.4824219, 728.7455444, -337.5020447, 827.6334229, -1124.1157227, 1066.2475586

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2939959, upper bound: 853.2896504
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2939959, upper bound: 853.2896504
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -166.4912109, 739.1508179, -901.2483521, 889.1364136
1: -202.9634247, 822.4923096, -208.2427063, 841.4191284, -1044.3825684, 1030.7348633
2: -207.2748718, 814.4534302, -212.4807587, 833.1720581, -1040.4468994, 1026.9342041
3: -328.6972656, 854.5055542, -336.8959045, 874.5217285, -1203.2187500, 1191.4014893
4: -331.8298645, 814.7550659, -339.9353027, 833.6307983, -1165.4603271, 1154.6900635

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2933772, upper bound: 853.2900110
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2933772, upper bound: 853.2900110
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -157.8990173, 700.5250244, -845.3319092, 804.8526001
1: -181.2450409, 736.1990356, -197.3341217, 797.1621094, -978.4071655, 933.5329590
2: -185.1202850, 728.8364258, -201.0137177, 789.2255859, -974.3458862, 929.8501587
3: -293.8229980, 764.5282593, -319.5162048, 828.4740601, -1122.2969971, 1084.0444336
4: -296.4824219, 728.7455444, -321.8711243, 789.2375488, -1085.7198486, 1050.6165771

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2943679
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2973834, upper bound: 853.2956453
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -171.5457764, 762.1805420, -906.9873657, 818.4993286
1: -181.2450409, 736.1990356, -214.5125122, 867.4678955, -1048.7127686, 950.7115479
2: -185.1202850, 728.8364258, -218.6978760, 858.8092041, -1043.9294434, 947.5343018
3: -293.8229980, 764.5282593, -347.3695679, 901.6154175, -1195.4382324, 1111.8975830
4: -296.4824219, 728.7455444, -350.2860107, 859.0719604, -1155.5541992, 1079.0314941

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2943679
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2956453
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -162.0976410, 722.6452026, -170.7339478, 759.0927734, -921.1902466, 893.3790894
1: -202.9634247, 822.4923096, -213.5276031, 863.9444580, -1066.9078369, 1036.0198975
2: -207.2748718, 814.4534302, -217.7257538, 855.3594360, -1062.6342773, 1032.1791992
3: -328.6972656, 854.5055542, -345.8427429, 897.9061890, -1226.6031494, 1200.3482666
4: -331.8298645, 814.7550659, -348.7878418, 855.6179810, -1187.4475098, 1163.5426025

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2798754, upper bound: 853.2630393
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2798754, upper bound: 853.2630393
time: 0.88 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3016558, upper bound: 853.3013192
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3021848, upper bound: 853.3017866
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3001867
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3001867
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3013621
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2981616, upper bound: 853.2981366
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2981616, upper bound: 853.2992671
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2983844, upper bound: 853.2984253
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2983844, upper bound: 853.2995889
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3008786
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3015926
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3008786
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3023444, upper bound: 853.3015926
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3003339, upper bound: 853.3003339
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3003339, upper bound: 853.3006109
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3006109, upper bound: 853.3007242
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3006109, upper bound: 853.3009987
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3001950
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3002044
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2983370
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2998001
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2983370
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2941821, upper bound: 853.2936496
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2978905, upper bound: 853.2959271
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2941821, upper bound: 853.2936496
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2978905, upper bound: 853.2959271
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2798728, upper bound: 853.2633247
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2798728, upper bound: 853.2633247
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3016675, upper bound: 853.3008962
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3014921, upper bound: 853.3007124
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2985446, upper bound: 853.2991498
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2985446, upper bound: 853.2991498
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2974063, upper bound: 853.2978749
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2974063, upper bound: 853.2978749
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2939959, upper bound: 853.2896504
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2939959, upper bound: 853.2896504
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2933772, upper bound: 853.2900110
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2933772, upper bound: 853.2900110
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2943679
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2973834, upper bound: 853.2956453
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2943679
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2956453
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2798754, upper bound: 853.2630393
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.52
Output dim: 0, lower bound: -853.2798754, upper bound: 853.2630393

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -134.0090179, 599.6547241, -153.2341461, 682.8717651, -816.8806152, 752.8887939
1: -167.7749939, 682.5620117, -191.8938904, 777.4130249, -945.1879883, 874.4558716
2: -171.6572723, 675.8182983, -196.2630310, 770.0264282, -941.6837158, 872.0812988
3: -271.8343506, 708.6510620, -310.4435730, 807.5858765, -1079.4201660, 1019.0946045
4: -274.6115723, 675.8950195, -313.7052002, 770.5253906, -1045.1369629, 989.6001587

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997579, upper bound: 853.2985766
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
time: 0.75 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.18 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.18
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.18
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3011437, upper bound: 853.2998046
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3016558, upper bound: 853.3013192
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3021848, upper bound: 853.3017866
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3001867
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3013621, upper bound: 853.3013621
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3001867
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992699, upper bound: 853.3013621
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3011631, upper bound: 853.2997989
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3013934, upper bound: 853.3001209
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2981616, upper bound: 853.2981366
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2981616, upper bound: 853.2992671
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2983844, upper bound: 853.2984253
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2983844, upper bound: 853.2995889
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3002049
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2992671, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2995889, upper bound: 853.3004331
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3008786
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3015926
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3014624, upper bound: 853.3008786
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3023444, upper bound: 853.3015926
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3003339, upper bound: 853.3003339
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3003339, upper bound: 853.3006109
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3006109, upper bound: 853.3007242
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3006109, upper bound: 853.3009987
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3000862, upper bound: 853.2990257
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3001950
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3009468, upper bound: 853.3002044
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2983370
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2998001
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2978560, upper bound: 853.2983370
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3000050, upper bound: 853.2998001
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2978905, upper bound: 853.2959271
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2978905, upper bound: 853.2959271
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3004739, upper bound: 853.2988928
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3016675, upper bound: 853.3008962
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3014921, upper bound: 853.3007124
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2985446, upper bound: 853.2991498
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2985446, upper bound: 853.2991498
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2998001, upper bound: 853.3000050
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2998607, upper bound: 853.2983145
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.3007702, upper bound: 853.2996936
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2974063, upper bound: 853.2978749
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2974063, upper bound: 853.2978749
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2987451, upper bound: 853.2987451
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956327, upper bound: 853.2969516
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2946302, upper bound: 853.2960073
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2943679
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2973834, upper bound: 853.2956453
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2943679
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 0, lower bound: -853.2956347, upper bound: 853.2956453
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024682, upper bound: 853.3025040
time: 0.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023290, upper bound: 853.3023290
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -853.3024682, upper bound: 853.3025040
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -853.3023290, upper bound: 853.3023290

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -179.2388916, 793.9887695, -181.6804962, 804.2030640, -983.4419556, 975.6692505
1: -224.1477814, 903.8054199, -227.1828918, 915.4710693, -1139.6188965, 1130.9882812
2: -228.6609802, 895.1587524, -231.7503662, 906.6832275, -1135.3441162, 1126.9091797
3: -362.4073792, 939.4089966, -367.2089844, 951.5463867, -1313.9536133, 1306.6179199
4: -365.4209595, 895.6079712, -370.3013306, 907.1323853, -1272.5533447, 1265.9093018

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024679, upper bound: 853.3024633
time: 0.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021712, upper bound: 853.3022832
time: 0.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -183.7534332, 814.8997803, -180.7377930, 800.4979248, -984.2513428, 995.6375732
1: -229.7754364, 927.4383545, -226.0026398, 911.2310181, -1141.0064697, 1153.4410400
2: -234.2279053, 918.4588013, -230.5442505, 902.4570312, -1136.6849365, 1149.0030518
3: -371.8710327, 963.9637451, -365.3877869, 947.1156616, -1318.9865723, 1329.3515625
4: -374.7720337, 918.7300415, -368.4625854, 902.8771362, -1277.6491699, 1287.1926270

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019190, upper bound: 853.3020545
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017622, upper bound: 853.3017622
time: 0.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -853.3024679, upper bound: 853.3024633
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -853.3021712, upper bound: 853.3022832
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -853.3019190, upper bound: 853.3020545
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -853.3017622, upper bound: 853.3017622

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -176.0336761, 780.3094482, -160.2791443, 712.9115601, -888.9452515, 940.5885620
1: -220.1692047, 888.2448730, -200.6800842, 811.6365356, -1031.8057861, 1088.9248047
2: -224.6620026, 879.6929321, -205.1961823, 803.9598389, -1028.6218262, 1084.8889160
3: -356.0254517, 923.2037964, -324.4720459, 843.1898193, -1199.2153320, 1247.6757812
4: -359.0802612, 880.0966797, -327.8054504, 804.4866333, -1163.5667725, 1207.9018555

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
time: 0.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -175.6157684, 778.2842407, -173.8045349, 770.0826416, -945.6984253, 952.0887451
1: -219.6191254, 885.9293823, -217.3294067, 876.6350708, -1096.2541504, 1103.2587891
2: -224.0371552, 877.3842773, -221.6949310, 868.0720215, -1092.1091309, 1099.0791016
3: -355.1373596, 920.8489380, -351.3989258, 911.2131958, -1266.3503418, 1272.2478027
4: -358.1421204, 877.8662720, -354.4614868, 868.5652466, -1226.7072754, 1232.3277588

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955601, upper bound: 853.2945379
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2940752, upper bound: 853.2935857
time: 0.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -180.8898010, 802.6335449, -159.3647919, 709.2182617, -890.1080322, 961.9981689
1: -226.2073059, 913.4955444, -199.5401764, 807.4058838, -1033.6131592, 1113.0356445
2: -230.6433105, 904.5766602, -204.0227051, 799.7489624, -1030.3923340, 1108.5993652
3: -366.1447449, 949.4369507, -322.6991882, 838.7740479, -1204.9185791, 1272.1361084
4: -369.1163635, 904.7756958, -325.9927979, 800.2584229, -1169.3746338, 1230.7685547

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977688, upper bound: 853.2970109
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2934070, upper bound: 853.2945337
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -179.9792938, 798.6517334, -172.7931519, 766.1118164, -946.0911255, 971.4448242
1: -225.0598755, 908.9491577, -216.0611877, 872.0938721, -1097.1534424, 1125.0102539
2: -229.4220428, 900.0613403, -220.4028473, 863.5437622, -1092.9658203, 1120.4642334
3: -364.3164673, 944.7548828, -349.4425049, 906.4648438, -1270.7812500, 1294.1973877
4: -367.2121887, 900.3551025, -352.4863281, 864.0028687, -1231.2150879, 1252.8414307

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955786, upper bound: 853.2943667
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2919969, upper bound: 853.2919969
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.89 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.2955601, upper bound: 853.2945379
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.2940752, upper bound: 853.2935857
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.2977688, upper bound: 853.2970109
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.2934070, upper bound: 853.2945337
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.2955786, upper bound: 853.2943667
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.89
Output dim: 0, lower bound: -853.2919969, upper bound: 853.2919969

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -170.6471252, 756.1704102, -158.9958801, 707.1309814, -877.7778931, 915.1662598
1: -213.3514404, 860.8896484, -199.0586548, 805.0988770, -1018.4501343, 1059.9481201
2: -217.7105408, 852.4254150, -203.5451202, 797.4287720, -1015.1392822, 1055.9705811
3: -345.0483093, 894.7868042, -321.8538513, 836.4036865, -1181.4519043, 1216.6405029
4: -348.2329407, 852.6338501, -325.2294312, 797.9222412, -1146.1551514, 1177.8632812

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -172.4749451, 765.8087769, -158.7764893, 706.1322632, -878.6071167, 924.5850830
1: -215.7381439, 871.6525879, -198.7887421, 803.9381714, -1019.6763306, 1070.4412842
2: -220.1859741, 863.2434692, -203.2660522, 796.3183594, -1016.5043335, 1066.5095215
3: -349.0492859, 905.7944336, -321.3940430, 835.2005615, -1184.2498779, 1227.1881104
4: -351.9735107, 863.6262207, -324.7422180, 796.8323364, -1148.8059082, 1188.3684082

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -156.3368530, 693.6622925, -170.1927338, 754.2881470, -910.6248169, 863.8550415
1: -195.4259491, 789.4722290, -212.8179626, 858.6729736, -1054.0988770, 1002.2901001
2: -199.2678680, 781.7581787, -217.0962830, 850.2463379, -1049.5141602, 998.8544312
3: -316.2294617, 820.3961792, -344.1231079, 892.5524292, -1208.7817383, 1164.5192871
4: -318.6985168, 781.9108276, -347.2210999, 850.7099609, -1169.4084473, 1129.1318359

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955601, upper bound: 853.2945379
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955601, upper bound: 853.2945379
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -162.9546051, 722.6859131, -156.2482452, 695.5319824, -858.4865112, 878.9338989
1: -203.7011261, 822.3861084, -195.6454163, 791.8345947, -995.5356445, 1018.0314331
2: -207.5210876, 814.3269653, -200.0390015, 784.3029785, -991.8240356, 1014.3659668
3: -329.7560730, 854.6394043, -316.4137573, 822.6086426, -1152.3646240, 1171.0532227
4: -332.1175537, 814.2910156, -319.7048950, 784.8107910, -1116.9283447, 1133.9958496

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -176.9361877, 785.4706421, -158.0679169, 703.4998169, -880.4359741, 943.5385742
1: -221.2817993, 893.9817505, -197.9214325, 800.9055176, -1022.1872559, 1091.9031982
2: -225.6391144, 885.1900024, -202.3695984, 793.2938232, -1018.9329224, 1087.5595703
3: -358.2001953, 929.1535645, -320.0764465, 832.0263062, -1190.2264404, 1249.2299805
4: -361.1905823, 885.4110107, -323.3633118, 793.8185425, -1155.0089111, 1208.7742920

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2598951, upper bound: 853.2722772
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2595133, upper bound: 853.2707291
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -161.5194397, 716.3763428, -169.2606354, 750.6438599, -912.1632080, 885.6369629
1: -201.8839111, 815.1983032, -211.6501770, 854.5028076, -1056.3865967, 1026.8485107
2: -205.6503601, 807.1585693, -215.9070435, 846.0886230, -1051.7390137, 1023.0656128
3: -326.8366699, 847.2095337, -342.3241882, 888.1918335, -1215.0284424, 1189.5335693
4: -329.1909485, 807.1807251, -345.4042053, 846.5208130, -1175.7114258, 1152.5845947

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955786, upper bound: 853.2943667
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955786, upper bound: 853.2943667
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2955601, upper bound: 853.2945379
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2955601, upper bound: 853.2945379
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2598951, upper bound: 853.2722772
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2595133, upper bound: 853.2707291
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2955786, upper bound: 853.2943667
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 0, lower bound: -853.2955786, upper bound: 853.2943667

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -170.6471252, 756.1704102, -156.6350555, 697.4343262, -868.0812988, 912.8053589
1: -213.3514404, 860.8896484, -196.1184998, 794.0266113, -1007.3779297, 1057.0080566
2: -217.7105408, 852.4254150, -200.5573730, 786.4363403, -1004.1468506, 1052.9827881
3: -345.0483093, 894.7868042, -317.2423401, 824.8671875, -1169.9155273, 1212.0289307
4: -348.2329407, 852.6338501, -320.5661621, 786.8939209, -1135.1268311, 1173.1999512

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -170.6471252, 756.1704102, -164.0262299, 730.9522705, -901.5992432, 920.1966553
1: -213.3514404, 860.8896484, -205.3510742, 831.9671631, -1045.3186035, 1066.2407227
2: -217.7105408, 852.4254150, -209.7105560, 823.8272095, -1041.5377197, 1062.1359863
3: -345.0483093, 894.7868042, -332.5710754, 864.3488770, -1209.3972168, 1227.3577881
4: -348.2329407, 852.6338501, -335.7479858, 824.0697632, -1172.3026123, 1188.3818359

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -172.4749451, 765.8087769, -156.4889832, 696.7344971, -869.2094116, 922.2976685
1: -215.7381439, 871.6525879, -195.9392090, 793.2070312, -1008.9451904, 1067.5917969
2: -220.1859741, 863.2434692, -200.3704376, 785.6653442, -1005.8513184, 1063.6138916
3: -349.0492859, 905.7944336, -316.9234924, 824.0173950, -1173.0665283, 1222.7177734
4: -351.9735107, 863.6262207, -320.2232056, 786.1436768, -1138.1170654, 1183.8493652

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3024152
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -172.4749451, 765.8087769, -163.8000793, 730.0144653, -902.4893799, 929.6088867
1: -215.7381439, 871.6525879, -205.0793304, 830.8737183, -1046.6118164, 1076.7318115
2: -220.1859741, 863.2434692, -209.4332428, 822.7830811, -1042.9688721, 1072.6767578
3: -349.0492859, 905.7944336, -332.1163330, 863.2127686, -1212.2617188, 1237.9104004
4: -351.9735107, 863.6262207, -335.2715454, 823.0469971, -1175.0205078, 1198.8977051

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -156.3368530, 693.6622925, -167.8423004, 744.4949951, -900.8316650, 861.5045776
1: -195.4259491, 789.4722290, -209.8958588, 847.4915771, -1042.9174805, 999.3681030
2: -199.2678680, 781.7581787, -214.1233978, 839.1918335, -1038.4597168, 995.8815918
3: -316.2294617, 820.3961792, -339.5009155, 880.9121704, -1197.1412354, 1159.8970947
4: -318.6985168, 781.9108276, -342.5396423, 839.6380615, -1158.3365479, 1124.4504395

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -156.3368530, 693.6622925, -172.2521820, 765.0521240, -921.3887329, 865.9143677
1: -195.4259491, 789.4722290, -215.3899078, 870.7236328, -1066.1495361, 1004.8621216
2: -199.2678680, 781.7581787, -219.5774231, 862.0755615, -1061.3433838, 1001.3355713
3: -316.2294617, 820.3961792, -348.7593079, 905.0347290, -1221.2641602, 1169.1555176
4: -318.6985168, 781.9108276, -351.6997986, 862.3270874, -1181.0256348, 1133.6105957

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -162.9546051, 722.6859131, -154.8936157, 689.8815918, -852.8361206, 877.5794678
1: -203.7011261, 822.3861084, -193.9588928, 785.3931885, -989.0942383, 1016.3448486
2: -207.5210876, 814.3269653, -198.3404694, 777.9203491, -985.4414062, 1012.6674194
3: -329.7560730, 854.6394043, -313.7467651, 815.9071655, -1145.6630859, 1168.3862305
4: -332.1175537, 814.2910156, -317.0288696, 778.4080200, -1110.5256348, 1131.3198242

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2967983, upper bound: 853.2967397
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -162.9546051, 722.6859131, -162.1234436, 722.9006348, -885.8552246, 884.8093262
1: -203.7011261, 822.3861084, -202.9975128, 822.7719727, -1026.4730225, 1025.3835449
2: -207.5210876, 814.3269653, -207.3012695, 814.7509766, -1022.2720337, 1021.6281738
3: -329.7560730, 854.6394043, -328.7904053, 854.7924805, -1184.5483398, 1183.4298096
4: -332.1175537, 814.2910156, -331.9281311, 815.0171509, -1147.1347656, 1146.2191162

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2967983, upper bound: 853.2967397
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -161.5194397, 716.3763428, -167.8423004, 744.4949951, -906.0144043, 884.2186279
1: -201.8839111, 815.1983032, -209.8958588, 847.4915771, -1049.3753662, 1025.0941162
2: -205.6503601, 807.1585693, -214.1233978, 839.1918335, -1044.8420410, 1021.2819824
3: -326.8366699, 847.2095337, -339.5009155, 880.9121704, -1207.7487793, 1186.7104492
4: -329.1909485, 807.1807251, -342.5396423, 839.6380615, -1168.8287354, 1149.7203369

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -161.5194397, 716.3763428, -172.2521820, 765.0521240, -926.5714722, 888.6284180
1: -201.8839111, 815.1983032, -215.3899078, 870.7236328, -1072.6075439, 1030.5882568
2: -205.6503601, 807.1585693, -219.5774231, 862.0755615, -1067.7258301, 1026.7359619
3: -326.8366699, 847.2095337, -348.7593079, 905.0347290, -1231.8713379, 1195.9688721
4: -329.1909485, 807.1807251, -351.6997986, 862.3270874, -1191.5177002, 1158.8804932

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.86 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3016544
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3015240, upper bound: 853.3024152
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.3023622, upper bound: 853.3024152
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.86
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -152.6555481, 679.6038208, -156.6350555, 697.4343262, -850.0898438, 836.2387695
1: -191.0909882, 773.8436279, -196.1184998, 794.0266113, -985.1176147, 969.9620361
2: -195.4390106, 766.2788696, -200.5573730, 786.4363403, -981.8753052, 966.8361816
3: -309.1385193, 803.9011841, -317.2423401, 824.8671875, -1134.0057373, 1121.1433105
4: -312.5699768, 766.6260986, -320.5661621, 786.8939209, -1099.4638672, 1087.1921387

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998391, upper bound: 853.3008936
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992311, upper bound: 853.3001303
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -166.0833435, 736.2805786, -156.6350555, 697.4343262, -863.5175781, 892.9155273
1: -207.6091461, 838.2443237, -196.1184998, 794.0266113, -1001.6357422, 1034.3627930
2: -211.7997131, 829.8923340, -200.5573730, 786.4363403, -998.2360229, 1030.4495850
3: -335.8413696, 871.2979736, -317.2423401, 824.8671875, -1160.7084961, 1188.5400391
4: -338.9661865, 830.1865845, -320.5661621, 786.8939209, -1125.8601074, 1150.7526855

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998391, upper bound: 853.3008936
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992311, upper bound: 853.3001303
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -152.6555481, 679.6038208, -164.0262299, 730.9522705, -883.6077881, 843.6300659
1: -191.0909882, 773.8436279, -205.3510742, 831.9671631, -1023.0581665, 979.1946411
2: -195.4390106, 766.2788696, -209.7105560, 823.8272095, -1019.2662354, 975.9893799
3: -309.1385193, 803.9011841, -332.5710754, 864.3488770, -1173.4874268, 1136.4722900
4: -312.5699768, 766.6260986, -335.7479858, 824.0697632, -1136.6396484, 1102.3740234

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3012816
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3016544
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -166.0833435, 736.2805786, -164.0262299, 730.9522705, -897.0355835, 900.3068237
1: -207.6091461, 838.2443237, -205.3510742, 831.9671631, -1039.5761719, 1043.5954590
2: -211.7997131, 829.8923340, -209.7105560, 823.8272095, -1035.6268311, 1039.6029053
3: -335.8413696, 871.2979736, -332.5710754, 864.3488770, -1200.1901855, 1203.8688965
4: -338.9661865, 830.1865845, -335.7479858, 824.0697632, -1163.0357666, 1165.9345703

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3012816
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3016544
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -154.5791779, 689.2937622, -156.4889832, 696.7344971, -851.3136597, 845.7825317
1: -193.5498352, 784.6648560, -195.9392090, 793.2070312, -986.7566528, 980.6040649
2: -197.9729462, 777.1978149, -200.3704376, 785.6653442, -983.6383057, 977.5682373
3: -313.2518921, 814.9785156, -316.9234924, 824.0173950, -1137.2691650, 1131.9019775
4: -316.4134827, 777.6448364, -320.2232056, 786.1436768, -1102.5571289, 1097.8680420

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014315, upper bound: 853.3020726
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012908, upper bound: 853.3020478
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -167.0906830, 742.3508911, -156.4889832, 696.7344971, -863.8250122, 898.8397217
1: -208.9735718, 844.9582520, -195.9392090, 793.2070312, -1002.1806030, 1040.8973389
2: -213.2379456, 836.6649780, -200.3704376, 785.6653442, -998.9031982, 1037.0352783
3: -338.1792603, 878.1000366, -316.9234924, 824.0173950, -1162.1966553, 1195.0234375
4: -341.0626831, 837.1320801, -320.2232056, 786.1436768, -1127.2062988, 1157.3552246

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014315, upper bound: 853.3020726
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012908, upper bound: 853.3020478
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -154.5791779, 689.2937622, -163.8000793, 730.0144653, -884.5936279, 853.0938721
1: -193.5498352, 784.6648560, -205.0793304, 830.8737183, -1024.4233398, 989.7442017
2: -197.9729462, 777.1978149, -209.4332428, 822.7830811, -1020.7560425, 986.6309814
3: -313.2518921, 814.9785156, -332.1163330, 863.2127686, -1176.4643555, 1147.0948486
4: -316.4134827, 777.6448364, -335.2715454, 823.0469971, -1139.4604492, 1112.9163818

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009111, upper bound: 853.3007447
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023495, upper bound: 853.3023828
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3019738
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3024152
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -167.0906830, 742.3508911, -163.8000793, 730.0144653, -897.1049805, 906.1509399
1: -208.9735718, 844.9582520, -205.0793304, 830.8737183, -1039.8472900, 1050.0374756
2: -213.2379456, 836.6649780, -209.4332428, 822.7830811, -1036.0208740, 1046.0982666
3: -338.1792603, 878.1000366, -332.1163330, 863.2127686, -1201.3918457, 1210.2163086
4: -341.0626831, 837.1320801, -335.2715454, 823.0469971, -1164.1096191, 1172.4035645

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009111, upper bound: 853.3007447
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023495, upper bound: 853.3023828
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3019738
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3024152
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -150.8036041, 669.6978760, -166.4879913, 738.4031982, -889.2067871, 836.1857910
1: -188.4811707, 762.2412109, -208.1965332, 840.5648193, -1029.0458984, 970.4377441
2: -192.1990662, 754.7455444, -212.3953094, 832.3268433, -1024.5258789, 967.1408691
3: -305.1082153, 792.0822754, -336.7425232, 873.7305298, -1178.8387451, 1128.8243408
4: -307.6214905, 754.8480225, -339.7876587, 832.7680054, -1140.3894043, 1094.6354980

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998317, upper bound: 853.2991317
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000651, upper bound: 853.2993787
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -147.9462891, 656.3718262, -163.8132935, 726.5757446, -874.5220337, 820.1849365
1: -184.8897400, 747.0568848, -204.8227539, 827.1246338, -1012.0144043, 951.8795776
2: -188.5782013, 739.6675415, -208.9729462, 818.9524536, -1007.5306396, 948.6404419
3: -299.1853027, 776.3476562, -331.3114319, 859.7592163, -1158.9444580, 1107.6590576
4: -301.7125549, 739.8026123, -334.3953857, 819.3625488, -1121.0750732, 1074.1979980

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987840, upper bound: 853.2985294
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2990201, upper bound: 853.2987730
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -150.8036041, 669.6978760, -170.8378143, 758.7368774, -909.5404053, 840.5357056
1: -188.4811707, 762.2412109, -213.6173248, 863.5425415, -1052.0235596, 975.8585205
2: -192.1990662, 754.7455444, -217.7742615, 854.9578857, -1047.1566162, 972.5197754
3: -305.1082153, 792.0822754, -345.8897705, 897.5883179, -1202.6964111, 1137.9719238
4: -307.6214905, 754.8480225, -348.8339233, 855.2066650, -1162.8281250, 1103.6817627

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
time: 1.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -147.9462891, 656.3718262, -168.2852478, 747.3594971, -895.3057861, 824.6569824
1: -184.8897400, 747.0568848, -210.3954620, 850.6143188, -1035.5040283, 957.4522705
2: -188.5782013, 739.6675415, -214.5053253, 842.0931396, -1030.6713867, 954.1727905
3: -299.1853027, 776.3476562, -340.6903687, 884.1495972, -1183.3348389, 1117.0380859
4: -301.7125549, 739.8026123, -343.6736145, 842.3012695, -1144.0137939, 1083.4761963

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -154.8936157, 689.8815918, -834.6884766, 801.8471069
1: -181.2450409, 736.1990356, -193.9588928, 785.3931885, -966.6382446, 930.1576538
2: -185.1202850, 728.8364258, -198.3404694, 777.9203491, -963.0406494, 927.1768799
3: -293.8229980, 764.5282593, -313.7467651, 815.9071655, -1109.7301025, 1078.2750244
4: -296.4824219, 728.7455444, -317.0288696, 778.4080200, -1074.8902588, 1045.7744141

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -157.8990173, 700.5250244, -154.8936157, 689.8815918, -847.7805786, 855.4185181
1: -197.3341217, 797.1621094, -193.9588928, 785.3931885, -982.7272339, 991.1208496
2: -201.0137177, 789.2255859, -198.3404694, 777.9203491, -978.9340820, 987.5660400
3: -319.5162048, 828.4740601, -313.7467651, 815.9071655, -1135.4233398, 1142.2208252
4: -321.8711243, 789.2375488, -317.0288696, 778.4080200, -1100.2790527, 1106.2662354

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -144.8068848, 646.9536133, -162.1234436, 722.9006348, -867.7075195, 809.0770264
1: -181.2450409, 736.1990356, -202.9975128, 822.7719727, -1004.0170288, 939.1963501
2: -185.1202850, 728.8364258, -207.3012695, 814.7509766, -999.8711548, 936.1376953
3: -293.8229980, 764.5282593, -328.7904053, 854.7924805, -1148.6152344, 1093.3184814
4: -296.4824219, 728.7455444, -331.9281311, 815.0171509, -1111.4995117, 1060.6737061

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -157.8990173, 700.5250244, -162.1234436, 722.9006348, -880.7996216, 862.6484375
1: -197.3341217, 797.1621094, -202.9975128, 822.7719727, -1020.1059570, 1000.1594849
2: -201.0137177, 789.2255859, -207.3012695, 814.7509766, -1015.7647095, 996.5268555
3: -319.5162048, 828.4740601, -328.7904053, 854.7924805, -1174.3085938, 1157.2644043
4: -321.8711243, 789.2375488, -331.9281311, 815.0171509, -1136.8883057, 1121.1655273

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -156.0110779, 692.6298828, -166.4879913, 738.4031982, -894.4143066, 859.1177979
1: -194.9767151, 788.2245483, -208.1965332, 840.5648193, -1035.5413818, 996.4210815
2: -198.6152802, 780.3982544, -212.3953094, 832.3268433, -1030.9420166, 992.7935791
3: -315.7964783, 819.1747437, -336.7425232, 873.7305298, -1189.5269775, 1155.9171143
4: -318.1910095, 780.3735962, -339.7876587, 832.7680054, -1150.9589844, 1120.1610107

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961171, upper bound: 853.2945589
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960967, upper bound: 853.2945841
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -153.1785889, 679.2051392, -163.8132935, 726.5757446, -879.7543335, 843.0183105
1: -191.4071503, 772.9229736, -204.8227539, 827.1246338, -1018.5317383, 977.7457275
2: -195.0184784, 765.2171631, -208.9729462, 818.9524536, -1013.9709473, 974.1900024
3: -309.8790588, 803.3233032, -331.3114319, 859.7592163, -1169.6383057, 1134.6347656
4: -312.2979736, 765.2066650, -334.3953857, 819.3625488, -1131.6605225, 1099.6020508

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956639, upper bound: 853.2945538
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956673, upper bound: 853.2945841
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -156.0110779, 692.6298828, -170.8378143, 758.7368774, -914.7479248, 863.4676514
1: -194.9767151, 788.2245483, -213.6173248, 863.5425415, -1058.5191650, 1001.8418579
2: -198.6152802, 780.3982544, -217.7742615, 854.9578857, -1053.5727539, 998.1724854
3: -315.7964783, 819.1747437, -345.8897705, 897.5883179, -1213.3847656, 1165.0644531
4: -318.1910095, 780.3735962, -348.8339233, 855.2066650, -1173.3977051, 1129.2073975

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -153.1785889, 679.2051392, -168.2852478, 747.3594971, -900.5380859, 847.4903564
1: -191.4071503, 772.9229736, -210.3954620, 850.6143188, -1042.0214844, 983.3183594
2: -195.0184784, 765.2171631, -214.5053253, 842.0931396, -1037.1115723, 979.7223511
3: -309.8790588, 803.3233032, -340.6903687, 884.1495972, -1194.0286865, 1144.0136719
4: -312.2979736, 765.2066650, -343.6736145, 842.3012695, -1154.5992432, 1108.8802490

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.04 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2998391, upper bound: 853.3008936
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2992311, upper bound: 853.3001303
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2998391, upper bound: 853.3008936
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2992311, upper bound: 853.3001303
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3012816
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3016544
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3012816
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3010954, upper bound: 853.3016544
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3014315, upper bound: 853.3020726
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3012908, upper bound: 853.3020478
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3014315, upper bound: 853.3020726
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3012908, upper bound: 853.3020478
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3019738
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3024152
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3019738
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3018387, upper bound: 853.3024152
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2998317, upper bound: 853.2991317
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.3000651, upper bound: 853.2993787
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2987840, upper bound: 853.2985294
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2990201, upper bound: 853.2987730
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2955558, upper bound: 853.2945126
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2954593, upper bound: 853.2944551
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2975186
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2976390, upper bound: 853.2970109
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2961171, upper bound: 853.2945589
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2960967, upper bound: 853.2945841
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2956639, upper bound: 853.2945538
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2956673, upper bound: 853.2945841
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2955764, upper bound: 853.2943398
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 0, lower bound: -853.2950376, upper bound: 853.2941881

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -149.5262146, 665.8372803, -136.5342255, 610.1560669, -759.6822510, 802.3715210
1: -187.1789703, 758.1782227, -170.9031830, 694.5503540, -881.7293091, 929.0812988
2: -191.4379425, 750.7348633, -174.8222961, 687.6690063, -879.1068726, 925.5571289
3: -302.8179932, 787.6363525, -276.8369751, 721.1520386, -1023.9700317, 1064.4733887
4: -306.2503662, 751.0782471, -279.6595764, 687.7232666, -993.9735718, 1030.7377930

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994160, upper bound: 853.3005842
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994160, upper bound: 853.3018841
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -151.3446045, 673.8335571, -153.6269226, 684.1795654, -835.5241699, 827.4604492
1: -189.4556885, 767.2795410, -192.3643646, 778.9540405, -968.4097290, 959.6439209
2: -193.7680664, 759.7614746, -196.7195587, 771.4660034, -965.2340698, 956.4810181
3: -306.4910278, 797.0879517, -311.1623840, 809.2205811, -1115.7115479, 1108.2502441
4: -309.9130859, 760.1264648, -314.4674377, 771.9636841, -1081.8767090, 1074.5938721

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2989499, upper bound: 853.2998770
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2989499, upper bound: 853.3010101
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -162.3738251, 720.0850830, -136.5342255, 610.1560669, -772.5298462, 856.6193237
1: -202.9775696, 819.8282471, -170.9031830, 694.5503540, -897.5279541, 990.7313232
2: -207.0779114, 811.6143799, -174.8222961, 687.6690063, -894.7469482, 986.4366455
3: -328.3749695, 852.1610107, -276.8369751, 721.1520386, -1049.5269775, 1128.9980469
4: -331.5335083, 811.8826294, -279.6595764, 687.7232666, -1019.2567139, 1091.5422363

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2993994, upper bound: 853.3005933
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997368, upper bound: 853.3008658
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -164.4040833, 729.0520020, -153.6269226, 684.1795654, -848.5836182, 882.6789551
1: -205.5171967, 830.0253906, -192.3643646, 778.9540405, -984.4712524, 1022.3897705
2: -209.6718292, 821.7271118, -196.7195587, 771.4660034, -981.1377563, 1018.4466553
3: -332.4808655, 862.7414551, -311.1623840, 809.2205811, -1141.7012939, 1173.9036865
4: -335.6030884, 822.0216675, -314.4674377, 771.9636841, -1107.5664062, 1136.4891357

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2989404, upper bound: 853.2998906
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2992311, upper bound: 853.3001303
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -152.6555481, 679.6038208, -159.9538116, 712.6268921, -865.2824707, 839.5576172
1: -191.0909882, 773.8436279, -200.2038727, 811.2327881, -1002.3237915, 974.0473633
2: -195.4390106, 766.2788696, -204.4667358, 803.1159668, -998.5549927, 970.7454834
3: -309.1385193, 803.9011841, -324.2623901, 842.8153076, -1151.9536133, 1128.1635742
4: -312.5699768, 766.6260986, -327.5487061, 803.2375488, -1115.8074951, 1094.1746826

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014276, upper bound: 853.3016338
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014276, upper bound: 853.3016456
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -152.6555481, 679.6038208, -162.1290894, 723.0679932, -875.7235107, 841.7328491
1: -191.0909882, 773.8436279, -202.9609833, 822.9121704, -1014.0031738, 976.8045044
2: -195.4390106, 766.2788696, -207.3266296, 814.8694458, -1010.3084717, 973.6054077
3: -309.1385193, 803.9011841, -328.7836304, 854.8229980, -1163.9613037, 1132.6848145
4: -312.5699768, 766.6260986, -331.8811951, 815.1316528, -1127.7016602, 1098.5072021

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014276, upper bound: 853.3017774
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014276, upper bound: 853.3017888
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -166.0833435, 736.2805786, -159.9538116, 712.6268921, -878.7102051, 896.2343750
1: -207.6091461, 838.2443237, -200.2038727, 811.2327881, -1018.8419189, 1038.4479980
2: -211.7997131, 829.8923340, -204.4667358, 803.1159668, -1014.9156494, 1034.3590088
3: -335.8413696, 871.2979736, -324.2623901, 842.8153076, -1178.6566162, 1195.5603027
4: -338.9661865, 830.1865845, -327.5487061, 803.2375488, -1142.2037354, 1157.7353516

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946476, upper bound: 853.2972946
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009323, upper bound: 853.3011726
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -166.0833435, 736.2805786, -162.1290894, 723.0679932, -889.1513672, 898.4096069
1: -207.6091461, 838.2443237, -202.9609833, 822.9121704, -1030.5211182, 1041.2053223
2: -211.7997131, 829.8923340, -207.3266296, 814.8694458, -1026.6691895, 1037.2189941
3: -335.8413696, 871.2979736, -328.7836304, 854.8229980, -1190.6643066, 1200.0814209
4: -338.9661865, 830.1865845, -331.8811951, 815.1316528, -1154.0976562, 1162.0677490

Time for backsubstitution: 2.04 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=991.4861450195312
rel_dist={0: [-853.3029989672884, 853.3029989672882]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1106.19 seconds
