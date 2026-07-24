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
execution time: IAR + LP analysis = 1.90 + 2.01 = 3.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031189


# Binary Search by BASE starts (time budget: 1196.09 seconds, max iter: 100)

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
Binary search time: 90.63 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1105.46 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031189
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031189
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031189
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031189

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -161.3719482, 717.4226074, -182.7681427, 808.7180176, -970.0898438, 900.1907349
1: -202.0404663, 816.7867432, -228.5348969, 920.6278076, -1122.6680908, 1045.3216553
2: -206.5800781, 809.0697021, -233.1255188, 911.7817383, -1118.3618164, 1042.1951904
3: -326.6069031, 848.5523071, -369.3415222, 956.9161377, -1283.5228271, 1217.8936768
4: -329.9661560, 809.6101074, -372.4694519, 912.2335815, -1242.1997070, 1182.0794678

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024522, upper bound: 853.3022252
time: 0.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021384, upper bound: 853.3020702
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -174.8621674, 774.4973755, -182.7681427, 808.7180176, -983.5801392, 957.2655029
1: -218.6441650, 881.6770020, -228.5348969, 920.6278076, -1139.2718506, 1110.2119141
2: -223.0339050, 873.0563965, -233.1255188, 911.7817383, -1134.8153076, 1106.1818848
3: -353.4781799, 916.4602661, -369.3415222, 956.9161377, -1310.3942871, 1285.8017578
4: -356.5664062, 873.5581665, -372.4694519, 912.2335815, -1268.7995605, 1246.0273438

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
time: 0.75 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020702, upper bound: 853.3020702
time: 1.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -853.3024522, upper bound: 853.3022252
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -853.3021384, upper bound: 853.3020702
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 0, lower bound: -853.3020702, upper bound: 853.3020702

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -161.3719482, 717.4226074, -172.6161652, 763.3468018, -924.7184448, 890.0387573
1: -202.0404663, 816.7867432, -215.8617249, 868.9192505, -1070.9593506, 1032.6484375
2: -206.5800781, 809.0697021, -220.2556305, 860.7608643, -1067.3409424, 1029.3253174
3: -326.6069031, 848.5523071, -348.6884155, 903.1724243, -1229.7792969, 1197.2403564
4: -329.9661560, 809.6101074, -351.4360046, 861.5000000, -1191.4660645, 1161.0460205

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004787, upper bound: 853.3007997
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024508, upper bound: 853.3021826
time: 0.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -161.0051880, 715.8868408, -203.9094696, 903.2384644, -1064.2436523, 919.7962646
1: -201.5813141, 815.0443115, -254.9292297, 1028.1209717, -1229.7022705, 1069.9735107
2: -206.1162262, 807.3298950, -260.1297302, 1018.3677979, -1224.4840088, 1067.4594727
3: -325.8806152, 846.7395630, -412.1533813, 1068.4002686, -1394.2807617, 1258.8929443
4: -329.2483215, 807.8594360, -415.5236206, 1018.7225342, -1347.9708252, 1223.3830566

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007077, upper bound: 853.2998086
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
time: 0.75 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -182.7681427, 808.7180176, -973.3994141, 911.7757568
1: -205.9361420, 829.8331299, -228.5348969, 920.6278076, -1126.5638428, 1058.3680420
2: -210.1320648, 821.9001465, -233.1255188, 911.7817383, -1121.9138184, 1055.0256348
3: -332.7737427, 862.5795288, -369.3415222, 956.9161377, -1289.6896973, 1231.9207764
4: -335.4803162, 822.6964111, -372.4694519, 912.2335815, -1247.7136230, 1195.1658936

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.72 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -196.0472412, 868.7943115, -182.5141296, 807.6267090, -1003.6739502, 1051.3084717
1: -245.1055756, 988.8947754, -228.2170868, 919.3861084, -1164.4914551, 1217.1118164
2: -250.1186981, 979.4112549, -232.8047180, 910.5521851, -1160.6707764, 1212.2158203
3: -396.3183899, 1027.7099609, -368.8308411, 955.6243896, -1351.9427490, 1396.5407715
4: -399.6038513, 980.0239868, -371.9601746, 911.0014648, -1310.6053467, 1351.9840088

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006394
time: 0.68 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3004787, upper bound: 853.3007997
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3024508, upper bound: 853.3021826
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3007077, upper bound: 853.2998086
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006394
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.35
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -172.2046814, 761.4098511, -909.6317749, 827.4152222
1: -185.5556030, 746.0098877, -215.3487091, 866.7153931, -1052.2708740, 961.3585205
2: -189.6679535, 739.3981323, -219.7298431, 858.5938110, -1048.2617188, 959.1279297
3: -299.1638184, 775.3083496, -347.8350525, 900.8940430, -1200.0578613, 1123.1431885
4: -302.1930847, 740.4500122, -350.5675659, 859.3494263, -1161.5422363, 1091.0174561

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004787, upper bound: 853.3007997
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004787, upper bound: 853.3007997
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -161.2957306, 717.0905762, -172.6161652, 763.3468018, -924.6423950, 889.7067261
1: -201.9450073, 816.4095459, -215.8617249, 868.9192505, -1070.8640137, 1032.2712402
2: -206.4831696, 808.6948853, -220.2556305, 860.7608643, -1067.2440186, 1028.9505615
3: -326.4538574, 848.1599121, -348.6884155, 903.1724243, -1229.6262207, 1196.8480225
4: -329.8136902, 809.2348022, -351.4360046, 861.5000000, -1191.3135986, 1160.6706543

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012082, upper bound: 853.3011264
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -160.8781128, 715.3116455, -197.8976135, 871.8310547, -1032.7091064, 913.2092285
1: -201.4220276, 814.3900146, -247.3440247, 992.4859009, -1193.9079590, 1061.7338867
2: -205.9532471, 806.6822510, -252.2371979, 983.4658203, -1189.4190674, 1058.9193115
3: -325.6217957, 846.0618286, -399.0464478, 1031.4986572, -1357.1204834, 1245.1082764
4: -328.9888611, 807.2127686, -402.1404419, 983.9295654, -1312.9184570, 1209.3530273

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994650, upper bound: 853.2987524
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007077, upper bound: 853.2998086
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007077, upper bound: 853.2998086
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -161.0051880, 715.8868408, -203.8071899, 902.7960815, -1063.8011475, 919.6940308
1: -201.5813141, 815.0443115, -254.8004913, 1027.6179199, -1229.1992188, 1069.8448486
2: -206.1162262, 807.3298950, -259.9992065, 1017.8665771, -1223.9827881, 1067.3291016
3: -325.8806152, 846.7395630, -411.9476318, 1067.8769531, -1393.7573242, 1258.6872559
4: -329.2483215, 807.8594360, -415.3203125, 1018.2191772, -1347.4675293, 1223.1796875

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3009713
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -164.2560577, 727.0237427, -169.3082581, 745.0418091, -909.2978516, 896.3319702
1: -205.4060364, 827.5750732, -211.6505737, 848.1795654, -1053.5855713, 1039.2255859
2: -209.5894165, 819.6800537, -215.8709564, 840.5891113, -1050.1784668, 1035.5509033
3: -331.8940430, 860.2443237, -341.1656799, 881.8913574, -1213.7851562, 1201.4099121
4: -334.5872803, 820.4936523, -344.0336914, 841.3805542, -1175.9677734, 1164.5273438

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.71 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -182.6663055, 808.2723999, -972.9537964, 911.6739502
1: -205.9361420, 829.8331299, -228.4062958, 920.1218872, -1126.0578613, 1058.2393799
2: -210.1320648, 821.9001465, -232.9947968, 911.2778320, -1121.4097900, 1054.8948975
3: -332.7737427, 862.5795288, -369.1358643, 956.3899536, -1289.1633301, 1231.7153320
4: -335.4803162, 822.6964111, -372.2661743, 911.7257690, -1247.2058105, 1194.9626465

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -182.3929138, 807.0776978, -997.4982300, 1021.6738892
1: -238.0125122, 955.4159546, -228.0658417, 918.7611084, -1156.7735596, 1183.4815674
2: -242.7419586, 946.6467285, -232.6504974, 909.9348145, -1152.6767578, 1179.2972412
3: -384.0323486, 993.0491943, -368.5832214, 954.9761353, -1339.0085449, 1361.6324463
4: -387.0486145, 947.3689575, -371.7113342, 910.3853760, -1297.4337158, 1319.0803223

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006394
time: 0.76 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006394
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -182.5141296, 807.6267090, -1003.5778198, 1050.8939209
1: -244.9847565, 988.4234009, -228.2170868, 919.3861084, -1164.3708496, 1216.6400146
2: -249.9962616, 978.9415283, -232.8047180, 910.5521851, -1160.5482178, 1211.7462158
3: -396.1255798, 1027.2198486, -368.8308411, 955.6243896, -1351.7500000, 1396.0505371
4: -399.4131775, 979.5528564, -371.9601746, 911.0014648, -1310.4146729, 1351.5130615

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.80 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.91 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3004787, upper bound: 853.3007997
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3004787, upper bound: 853.3007997
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3012082, upper bound: 853.3011264
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3007077, upper bound: 853.2998086
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3007077, upper bound: 853.2998086
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006394
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006394
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -151.3268127, 672.2167969, -820.4388428, 806.5372314
1: -185.5556030, 746.0098877, -189.4799957, 765.2411499, -950.7967529, 935.4897461
2: -189.6679535, 739.3981323, -193.8190002, 758.2554321, -947.9234009, 933.2171631
3: -299.1638184, 775.3083496, -306.0753479, 794.9966431, -1094.1604004, 1081.3836670
4: -302.1930847, 740.4500122, -309.0246582, 759.0982666, -1061.2912598, 1049.4744873

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975439, upper bound: 853.2996478
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2966066
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -164.2457275, 726.9752808, -875.1972656, 819.4562378
1: -185.5556030, 746.0098877, -205.3932190, 827.5200195, -1013.0756226, 951.4030762
2: -189.6679535, 739.3981323, -209.5762634, 819.6260376, -1009.2940063, 948.9743652
3: -299.1638184, 775.3083496, -331.8727722, 860.1876221, -1159.3511963, 1107.1811523
4: -302.1930847, 740.4500122, -334.5655212, 820.4401245, -1122.6331787, 1075.0155029

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977580
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2966066
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -161.2957306, 717.0905762, -154.0258942, 681.4078369, -842.7034912, 871.1164551
1: -201.9450073, 816.4095459, -192.5411072, 775.5143433, -977.4593506, 1008.9506836
2: -206.4831696, 808.6948853, -196.3657684, 768.1785889, -974.6616821, 1005.0605469
3: -326.4538574, 848.1599121, -311.1072083, 805.9265137, -1132.3803711, 1159.2670898
4: -329.8136902, 809.2348022, -313.3131104, 768.6024780, -1098.4161377, 1122.5478516

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011972, upper bound: 853.3011264
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011654, upper bound: 853.3011264
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -161.2957306, 717.0905762, -168.9018097, 747.3297119, -908.6254272, 885.9923706
1: -201.9450073, 816.4095459, -211.2360840, 850.7069702, -1052.6518555, 1027.6456299
2: -206.4831696, 808.6948853, -215.5525818, 842.6710815, -1049.1539307, 1024.2474365
3: -326.4538574, 848.1599121, -341.2554626, 884.2236328, -1210.6773682, 1189.4152832
4: -329.8136902, 809.2348022, -344.0006714, 843.4244995, -1173.2381592, 1153.2354736

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -160.8781128, 715.3116455, -175.5885162, 776.4868164, -937.3648682, 890.9000244
1: -201.4220276, 814.3900146, -219.7106323, 883.9606934, -1085.3825684, 1034.1005859
2: -205.9532471, 806.6822510, -224.4729156, 876.0089111, -1081.9620361, 1031.1550293
3: -325.6217957, 846.0618286, -354.5040283, 918.4062500, -1244.0279541, 1200.5659180
4: -328.9888611, 807.2127686, -357.8605652, 876.7702637, -1205.7591553, 1165.0732422

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994650, upper bound: 853.2987524
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2984257
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2998086
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -160.8781128, 715.3116455, -190.4205170, 839.2810669, -1000.1591187, 905.7321777
1: -201.4220276, 814.3900146, -238.0125122, 955.4159546, -1156.8377686, 1052.4022217
2: -205.9532471, 806.6822510, -242.7419586, 946.6467285, -1152.5999756, 1049.4241943
3: -325.6217957, 846.0618286, -384.0323486, 993.0491943, -1318.6710205, 1230.0941162
4: -328.9888611, 807.2127686, -387.0486145, 947.3689575, -1276.3577881, 1194.2613525

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994650, upper bound: 853.2987524
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2984257
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2998086
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -161.0051880, 715.8868408, -180.1696320, 801.5511475, -962.5562134, 896.0563354
1: -201.5813141, 815.0443115, -225.4840698, 912.3982544, -1113.9793701, 1040.5283203
2: -206.1162262, 807.3298950, -230.5822754, 903.7421265, -1109.8583984, 1037.9121094
3: -325.8806152, 846.7395630, -364.6455078, 947.7800293, -1273.6604004, 1211.3850098
4: -329.2483215, 807.8594360, -368.4310913, 904.3006592, -1233.5489502, 1176.2905273

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3009713
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -161.0051880, 715.8868408, -195.9511719, 868.3797607, -1029.3850098, 911.8378906
1: -201.5813141, 815.0443115, -244.9847565, 988.4234009, -1190.0045166, 1060.0289307
2: -206.1162262, 807.3298950, -249.9962616, 978.9415283, -1185.0577393, 1057.3259277
3: -325.8806152, 846.7395630, -396.1255798, 1027.2198486, -1353.1000977, 1242.8651123
4: -329.2483215, 807.8594360, -399.4131775, 979.5528564, -1308.8011475, 1207.2725830

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3009713
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3019658
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -164.2457275, 726.9754028, -148.7031708, 657.2008057, -821.4465332, 875.6785278
1: -205.3932190, 827.5200195, -186.1483459, 748.2800903, -953.6732178, 1013.6683350
2: -209.5762634, 819.6260376, -190.2608185, 741.6431274, -951.2192993, 1009.8868408
3: -331.8727722, 860.1876221, -300.1105957, 777.6807251, -1109.5534668, 1160.2977295
4: -334.5655212, 820.4401245, -303.1169128, 742.7159424, -1077.2814941, 1123.5570068

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977532, upper bound: 853.2984038
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -164.2483521, 726.9868164, -161.5639496, 711.8400269, -876.0883789, 888.5507812
1: -205.3965759, 827.5331421, -201.9722748, 810.4206543, -1015.8171997, 1029.5053711
2: -209.5796509, 819.6390381, -206.0324402, 802.9714966, -1012.5511475, 1025.6715088
3: -331.8780823, 860.2011108, -325.6631470, 842.6292114, -1174.5073242, 1185.8642578
4: -334.5708618, 820.4531860, -328.5767822, 803.7938232, -1138.3647461, 1149.0297852

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977532, upper bound: 853.2984038
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
time: 1.06 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -161.2957306, 717.0905762, -881.7720337, 890.3033447
1: -205.9361420, 829.8331299, -201.9450073, 816.4095459, -1022.3457031, 1031.7780762
2: -210.1320648, 821.9001465, -206.4831696, 808.6948853, -1018.8269653, 1028.3831787
3: -332.7737427, 862.5795288, -326.4538574, 848.1599121, -1180.9334717, 1189.0332031
4: -335.4803162, 822.6964111, -329.8136902, 809.2348022, -1144.7149658, 1152.5101318

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991415, upper bound: 853.3006228
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 0.77 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -174.7723389, 774.1043701, -938.7858276, 903.7799683
1: -205.9361420, 829.8331299, -218.5308075, 881.2310181, -1087.1671143, 1048.3638916
2: -210.1320648, 821.9001465, -222.9185486, 872.6120605, -1082.7441406, 1044.8186035
3: -332.7737427, 862.5795288, -353.2970886, 915.9963989, -1248.7696533, 1215.8763428
4: -335.4803162, 822.6964111, -356.3869629, 873.1104126, -1208.5905762, 1179.0833740

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991415, upper bound: 853.3006228
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -160.8781128, 715.3116455, -905.7321777, 1000.1591187
1: -238.0125122, 955.4159546, -201.4220276, 814.3900146, -1052.4022217, 1156.8377686
2: -242.7419586, 946.6467285, -205.9532471, 806.6822510, -1049.4241943, 1152.5999756
3: -384.0323486, 993.0491943, -325.6217957, 846.0618286, -1230.0942383, 1318.6710205
4: -387.0486145, 947.3689575, -328.9888611, 807.2127686, -1194.2613525, 1276.3577881

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
time: 0.86 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006394
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -174.4824371, 772.8388672, -963.2593994, 1013.7634888
1: -238.0125122, 955.4159546, -218.1691742, 879.7891846, -1117.8013916, 1173.5847168
2: -242.7419586, 946.6467285, -222.5525208, 871.1881104, -1113.9300537, 1169.1992188
3: -384.0323486, 993.0491943, -352.7108154, 914.4982910, -1298.5306396, 1345.7600098
4: -387.0486145, 947.3689575, -355.7996216, 871.6879883, -1258.7365723, 1303.1682129

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
time: 0.84 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006394
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -161.0051880, 715.8868408, -911.8378906, 1029.3850098
1: -244.9847565, 988.4234009, -201.5813141, 815.0443115, -1060.0289307, 1190.0045166
2: -249.9962616, 978.9415283, -206.1162262, 807.3298950, -1057.3259277, 1185.0577393
3: -396.1255798, 1027.2198486, -325.8806152, 846.7395630, -1242.8651123, 1353.1000977
4: -399.4131775, 979.5528564, -329.2483215, 807.8594360, -1207.2725830, 1308.8011475

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015091
time: 0.73 seconds

## Relational analysis of IS_A2_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 1.27 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -174.6021118, 773.3803711, -969.3314209, 1042.9819336
1: -244.9847565, 988.4234009, -218.3187408, 880.4053955, -1125.3900146, 1206.7415771
2: -249.9962616, 978.9415283, -222.7050781, 871.7972412, -1121.7932129, 1201.6466064
3: -396.1255798, 1027.2198486, -352.9556274, 915.1378784, -1311.2634277, 1380.1749268
4: -399.4131775, 979.5528564, -356.0448608, 872.2962036, -1271.7093506, 1335.5976562

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.77 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.87 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2975439, upper bound: 853.2996478
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2966066
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977580
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2966066
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3011972, upper bound: 853.3011264
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3011654, upper bound: 853.3011264
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2984257
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2998086
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2984257
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2987356, upper bound: 853.2998086
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3019658
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020958, upper bound: 853.3020276
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006394
IS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006394
IS_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015091
IS_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.87
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -131.7478333, 586.7029419, -734.9249878, 786.9583740
1: -185.5556030, 746.0098877, -164.9117737, 667.7753296, -853.3308716, 910.9216309
2: -189.6679535, 739.3981323, -168.7386780, 661.4795532, -851.1475220, 908.1368408
3: -299.1638184, 775.3083496, -266.6155090, 693.4291992, -992.5930176, 1041.9238281
4: -302.1930847, 740.4500122, -269.1137390, 661.9461060, -964.1391602, 1009.5636597

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975329, upper bound: 853.2996478
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975010, upper bound: 853.2996478
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -148.3150330, 658.9260254, -807.1480103, 803.5255737
1: -185.5556030, 746.0098877, -185.7207031, 750.1322632, -935.6878662, 931.7305908
2: -189.6679535, 739.3981323, -189.9795074, 743.2528076, -932.9207764, 929.3776245
3: -299.1638184, 775.3083496, -299.9850769, 779.3118286, -1078.4755859, 1075.2933350
4: -302.1930847, 740.4500122, -302.9180298, 744.1358643, -1046.3286133, 1043.3677979

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2968235, upper bound: 853.2985340
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2965499, upper bound: 853.2984998
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -127.0278702, 563.1144409, -164.2297058, 726.9002075, -853.9280396, 727.3440552
1: -158.9505157, 641.1395264, -205.3732147, 827.4345093, -986.3850098, 846.5127563
2: -162.5271606, 635.0861816, -209.5557556, 819.5419922, -982.0690918, 844.6419678
3: -256.5744019, 666.0552979, -331.8395691, 860.0990601, -1116.6733398, 997.8947754
4: -259.2628784, 635.6389771, -334.5317078, 820.3568115, -1079.6192627, 970.1706543

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977580
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977580
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2917703, upper bound: 853.2844389
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979167, upper bound: 853.2976750
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -146.0648804, 645.6864014, -164.2384644, 726.9414062, -873.0062866, 809.9248047
1: -182.8547516, 735.1871948, -205.3841705, 827.4815063, -1010.3361816, 940.5713501
2: -186.9111023, 728.6476440, -209.5669708, 819.5881348, -1006.4991455, 938.2145996
3: -294.7907715, 764.0739136, -331.8577576, 860.1475830, -1154.9381104, 1095.9316406
4: -297.8164062, 729.6941528, -334.5502625, 820.4024048, -1118.2187500, 1064.2441406

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2966066
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2959309
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2921242, upper bound: 853.2927431
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2956152, upper bound: 853.2964686
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2952556, upper bound: 853.2951503
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957008, upper bound: 853.2963442
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -161.2957306, 717.0905762, -150.7081604, 667.2469482, -828.5426636, 867.7987061
1: -201.9450073, 816.4095459, -188.4238129, 759.3538818, -961.2988892, 1004.8332520
2: -206.4831696, 808.6948853, -192.2007599, 752.2247314, -958.7077637, 1000.8956299
3: -326.4538574, 848.1599121, -304.5221558, 789.1150513, -1115.5687256, 1152.6821289
4: -329.8136902, 809.2348022, -306.5414429, 752.7979126, -1082.6115723, 1115.7761230

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011972, upper bound: 853.3011264
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011972, upper bound: 853.3011264
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -160.8930817, 715.2909546, -159.3162689, 704.6448975, -865.5379639, 874.6071777
1: -201.4388123, 814.3654175, -199.1229706, 802.0049438, -1003.4437256, 1013.4884033
2: -205.9677277, 806.6665039, -203.1571808, 794.3808594, -1000.3485718, 1009.8236694
3: -325.6336060, 846.0399780, -321.7102966, 833.5355225, -1159.1691895, 1167.7502441
4: -328.9986572, 807.1987305, -324.1705322, 794.9080811, -1123.9067383, 1131.3692627

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011654, upper bound: 853.3011264
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011654, upper bound: 853.3011264
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -161.2957306, 717.0905762, -148.7417450, 660.8994751, -822.1951904, 865.8322754
1: -201.9450073, 816.4095459, -186.2520447, 752.3769531, -954.3219604, 1002.6615601
2: -206.4831696, 808.6948853, -190.5237122, 745.4625244, -951.9456787, 999.2186279
3: -326.4538574, 848.1599121, -300.8616028, 781.6331787, -1108.0869141, 1149.0213623
4: -329.8136902, 809.2348022, -303.8078918, 746.3312988, -1076.1450195, 1113.0427246

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -161.2957306, 717.0905762, -160.7426147, 712.0878296, -873.3835449, 877.8331299
1: -201.9450073, 816.4095459, -201.0252075, 810.5961304, -1012.5411377, 1017.4347534
2: -206.4831696, 808.6948853, -205.1365967, 802.7780762, -1009.2611084, 1013.8314819
3: -326.4538574, 848.1599121, -324.8952332, 842.5469360, -1169.0007324, 1173.0551758
4: -329.8136902, 809.2348022, -327.5987549, 803.5680542, -1133.3817139, 1136.8334961

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2920320, upper bound: 853.2844396
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979705, upper bound: 853.2973437
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -147.9601440, 653.9851074, -175.1508789, 774.3954468, -922.3554688, 829.1359863
1: -185.2247009, 744.6149902, -219.1609955, 881.5872192, -1066.8118896, 963.7759399
2: -189.3336945, 738.0192871, -223.9138336, 873.6697388, -1063.0032959, 961.9331055
3: -298.6232605, 773.8648682, -353.5883179, 915.9555664, -1214.5786133, 1127.4531250
4: -301.6665039, 739.0724487, -356.9451294, 874.4404297, -1176.1069336, 1096.0174561

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2958054, upper bound: 853.2975888
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946540, upper bound: 853.2946540
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -160.9203339, 715.5186768, -175.5885162, 776.4868164, -937.4069824, 891.1071167
1: -201.4749298, 814.6262207, -219.7106323, 883.9606934, -1085.4355469, 1034.3369141
2: -206.0082855, 806.9141235, -224.4729156, 876.0089111, -1082.0172119, 1031.3870850
3: -325.7101746, 846.3046265, -354.5040283, 918.4062500, -1244.1164551, 1200.8085938
4: -329.0786438, 807.4430542, -357.8605652, 876.7702637, -1205.8488770, 1165.3035889

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2958054, upper bound: 853.2990675
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946540, upper bound: 853.2961326
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -147.9601440, 653.9851074, -189.9815826, 837.2424927, -985.2025146, 843.9666748
1: -185.2247009, 744.6149902, -237.4679718, 953.1030884, -1138.3277588, 982.0829468
2: -189.3336945, 738.0192871, -242.1882019, 944.3597412, -1133.6934814, 980.2075195
3: -298.6232605, 773.8648682, -383.1297607, 990.6578979, -1289.2811279, 1156.9946289
4: -301.6665039, 739.0724487, -386.1362610, 945.1120605, -1246.7784424, 1125.2083740

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2958006, upper bound: 853.2972738
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987190, upper bound: 853.2982374
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977675, upper bound: 853.2976048
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983964, upper bound: 853.2982241
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2986983, upper bound: 853.2983928
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -160.9203339, 715.5186768, -190.4205170, 839.2810669, -1000.2012329, 905.9392090
1: -201.4749298, 814.6262207, -238.0125122, 955.4159546, -1156.8907471, 1052.6384277
2: -206.0082855, 806.9141235, -242.7419586, 946.6467285, -1152.6550293, 1049.6561279
3: -325.7101746, 846.3046265, -384.0323486, 993.0491943, -1318.7593994, 1230.3369141
4: -329.0786438, 807.4430542, -387.0486145, 947.3689575, -1276.4475098, 1194.4916992

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2958006, upper bound: 853.2987524
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961687, upper bound: 853.2984461
time: 0.88 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984374, upper bound: 853.2996665
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -180.1696320, 801.5511475, -953.3119507, 854.3991089
1: -190.0204926, 767.5302734, -225.4840698, 912.3982544, -1102.4185791, 993.0143433
2: -194.3727112, 760.5083618, -230.5822754, 903.7421265, -1098.1146240, 991.0906372
3: -306.9681091, 797.3639526, -364.6455078, 947.7800293, -1254.7481689, 1162.0095215
4: -309.9315796, 761.3359985, -368.4310913, 904.3006592, -1214.2321777, 1129.7668457

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3010395
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997969, upper bound: 853.2997969
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -180.2634430, 801.9572144, -180.1696320, 801.5511475, -981.8145752, 982.1267700
1: -225.6020966, 912.8598022, -225.4840698, 912.3982544, -1138.0001221, 1138.3438721
2: -230.7018127, 904.2016602, -230.5822754, 903.7421265, -1134.4437256, 1134.7838135
3: -364.8340759, 948.2597046, -364.6455078, 947.7800293, -1312.6138916, 1312.9049072
4: -368.6170349, 904.7614136, -368.4310913, 904.3006592, -1272.9177246, 1273.1923828

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3010392
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997969, upper bound: 853.2997969
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -158.1622467, 703.7423096, -195.9511719, 868.3797607, -1026.5419922, 899.6934204
1: -198.0095215, 801.2159424, -244.9847565, 988.4234009, -1186.4326172, 1046.2005615
2: -202.4909821, 793.6063232, -249.9962616, 978.9415283, -1181.4324951, 1043.6025391
3: -320.1865234, 832.3234253, -396.1255798, 1027.2198486, -1347.4058838, 1228.4489746
4: -323.5203247, 794.0815430, -399.4131775, 979.5528564, -1303.0732422, 1193.4947510

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3009713
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003186, upper bound: 853.3007118
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019674, upper bound: 853.3018264
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -161.3093414, 716.9349365, -195.6250000, 866.9166260, -1028.2259521, 912.5599365
1: -201.9517975, 816.2551880, -244.5750275, 986.7594604, -1188.7108154, 1060.8302002
2: -206.5014343, 808.5461426, -249.5800018, 977.2914429, -1183.7926025, 1058.1259766
3: -326.4016113, 848.0557251, -395.4592285, 1025.4921875, -1351.8937988, 1243.5146484
4: -329.8478699, 809.1218872, -398.7507324, 977.9010010, -1307.7487793, 1207.8725586

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017371, upper bound: 853.3017297
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -161.0787811, 713.4104614, -148.7031708, 657.2008057, -818.2794800, 862.1136475
1: -201.4603424, 812.0397949, -186.1483459, 748.2800903, -949.7402954, 998.1880493
2: -205.5950623, 804.3491211, -190.2608185, 741.6431274, -947.2381592, 994.6099243
3: -325.5805359, 844.0925293, -300.1105957, 777.6807251, -1103.2612305, 1144.2028809
4: -328.0908813, 805.3196411, -303.1169128, 742.7159424, -1070.8067627, 1108.4365234

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977580, upper bound: 853.2987189
time: 0.73 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2966066, upper bound: 853.2957841
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -168.8007812, 746.7077026, -148.4821472, 656.2284546, -825.0291748, 895.1898193
1: -211.0579681, 849.9569702, -185.8753357, 747.1715698, -958.2295532, 1035.8322754
2: -215.4025269, 841.8391724, -189.9824371, 740.5490112, -955.9514771, 1031.8216553
3: -340.9168091, 883.5670166, -299.6683960, 776.5288696, -1117.4456787, 1183.2353516
4: -343.7474976, 842.8374023, -302.6718140, 741.6289062, -1085.3762207, 1145.5092773

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996478, upper bound: 853.2975010
time: 0.73 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006207, upper bound: 853.3004342
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000880, upper bound: 853.2994851
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976023, upper bound: 853.2861533
time: 0.78 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007442, upper bound: 853.3003009
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -161.0813446, 713.4217529, -161.5639496, 711.8400269, -872.9213867, 874.9856567
1: -201.4636230, 812.0527344, -201.9722748, 810.4206543, -1011.8842773, 1014.0250244
2: -205.5983887, 804.3620605, -206.0324402, 802.9714966, -1008.5698853, 1010.3943481
3: -325.5857849, 844.1058350, -325.6631470, 842.6292114, -1168.2149658, 1169.7690430
4: -328.0961609, 805.3325195, -328.5767822, 803.7938232, -1131.8900146, 1133.9091797

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977532, upper bound: 853.2984038
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996070, upper bound: 853.2992113
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007442, upper bound: 853.3001090
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -168.8042145, 746.7243652, -161.3565826, 710.9173584, -879.7215576, 908.0809326
1: -211.0623169, 849.9759521, -201.7150421, 809.3698120, -1020.4321289, 1051.6909180
2: -215.4070282, 841.8578491, -205.7689514, 801.9298096, -1017.3368530, 1047.6268311
3: -340.9239197, 883.5864868, -325.2470398, 841.5394287, -1182.4632568, 1208.8334961
4: -343.7546692, 842.8562012, -328.1578979, 802.7611084, -1146.5157471, 1171.0139160

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007849, upper bound: 853.3000499
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006694, upper bound: 853.3000548
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -161.5120239, 715.4299927, -161.2957306, 717.0905762, -878.6024780, 876.7256470
1: -202.0000610, 814.3376465, -201.9450073, 816.4095459, -1018.4096069, 1016.2826538
2: -206.1473389, 806.6093750, -206.4831696, 808.6948853, -1014.8422241, 1013.0924072
3: -326.4761658, 846.4689941, -326.4538574, 848.1599121, -1174.6357422, 1172.9228516
4: -329.0003357, 807.5625000, -329.8136902, 809.2348022, -1138.2351074, 1137.3762207

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991415, upper bound: 853.3006910
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980852, upper bound: 853.2994484
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -169.2769623, 748.9935913, -160.8930817, 715.2909546, -884.5678711, 909.8866577
1: -211.6526184, 852.5535278, -201.4388123, 814.3654175, -1026.0179443, 1053.9920654
2: -216.0144806, 844.3994141, -205.9677277, 806.6665039, -1022.6809692, 1050.3671875
3: -341.9109802, 886.2436523, -325.6336060, 846.0399780, -1187.9509277, 1211.8771973
4: -344.7489624, 845.3828735, -328.9986572, 807.1987305, -1151.9477539, 1174.3814697

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011264, upper bound: 853.3011654
time: 0.78 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021208, upper bound: 853.3024080
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3024080
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -161.5120239, 715.4299927, -174.7723389, 774.1043701, -935.6162720, 890.2023315
1: -202.0000610, 814.3376465, -218.5308075, 881.2310181, -1083.2309570, 1032.8682861
2: -206.1473389, 806.6093750, -222.9185486, 872.6120605, -1078.7593994, 1029.5278320
3: -326.4761658, 846.4689941, -353.2970886, 915.9963989, -1242.4720459, 1199.7661133
4: -329.0003357, 807.5625000, -356.3869629, 873.1104126, -1202.1105957, 1163.9493408

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991415, upper bound: 853.3006228
time: 0.78 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -169.2769623, 748.9935913, -174.4796143, 772.7780151, -942.0549316, 923.4732056
1: -211.6526184, 852.5535278, -218.1643066, 879.7231445, -1091.3754883, 1070.7176514
2: -216.0144806, 844.3994141, -222.5452576, 871.1197510, -1087.1337891, 1066.9447021
3: -341.9109802, 886.2436523, -352.6978455, 914.4332275, -1256.3442383, 1238.9412842
4: -344.7489624, 845.3828735, -355.7887878, 871.6161499, -1216.3651123, 1201.1716309

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -189.9815826, 837.2424927, -147.9601440, 653.9851074, -843.9666748, 985.2025146
1: -237.4679718, 953.1030884, -185.2247009, 744.6149902, -982.0829468, 1138.3277588
2: -242.1882019, 944.3597412, -189.3336945, 738.0192871, -980.2075195, 1133.6934814
3: -383.1297607, 990.6578979, -298.6232605, 773.8648682, -1156.9946289, 1289.2811279
4: -386.1362610, 945.1120605, -301.6665039, 739.0724487, -1125.2084961, 1246.7784424

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2972738, upper bound: 853.2958006
time: 0.82 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982374, upper bound: 853.2987190
time: 0.76 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976048, upper bound: 853.2977675
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982241, upper bound: 853.2983964
time: 0.89 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983928, upper bound: 853.2986984
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -160.9203339, 715.5186768, -905.9392090, 1000.2012329
1: -238.0125122, 955.4159546, -201.4749298, 814.6262207, -1052.6384277, 1156.8907471
2: -242.7419586, 946.6467285, -206.0082855, 806.9141235, -1049.6561279, 1152.6550293
3: -384.0323486, 993.0491943, -325.7101746, 846.3046265, -1230.3369141, 1318.7593994
4: -387.0486145, 947.3689575, -329.0786438, 807.4430542, -1194.4916992, 1276.4475098

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2972738, upper bound: 853.2994650
time: 0.74 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2972080, upper bound: 853.2989454
time: 0.74 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983048, upper bound: 853.3005978
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -189.9944000, 837.2998047, -161.2456818, 710.3819580, -900.3763428, 998.5454712
1: -237.4837189, 953.1685181, -201.5750275, 808.7598267, -1046.2435303, 1154.7435303
2: -242.2044373, 944.4242554, -205.6291809, 801.3250732, -1043.5295410, 1150.0533447
3: -383.1552429, 990.7258301, -325.0190735, 840.9117432, -1224.0670166, 1315.7447510
4: -386.1624451, 945.1760254, -327.9354858, 802.1724243, -1188.3348389, 1273.1115723

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2982556
time: 0.82 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2982695
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -174.5123291, 772.9876099, -963.4081421, 1013.7933960
1: -238.0125122, 955.4159546, -218.2054596, 879.9597778, -1117.9720459, 1173.6213379
2: -242.7419586, 946.6467285, -222.5898285, 871.3530273, -1114.0949707, 1169.2364502
3: -384.0323486, 993.0491943, -352.7745972, 914.6742554, -1298.7065430, 1345.8237305
4: -387.0486145, 947.3689575, -355.8656311, 871.8488770, -1258.8974609, 1303.2344971

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2997722
time: 0.77 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2997861
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -193.2272644, 856.6335449, -161.0051880, 715.8868408, -909.1140137, 1017.6386108
1: -241.5540161, 975.0561523, -201.5813141, 815.0443115, -1056.5983887, 1176.6374512
2: -246.5111237, 965.6755371, -206.1162262, 807.3298950, -1053.8409424, 1171.7917480
3: -390.6423645, 1013.2908325, -325.8806152, 846.7395630, -1237.3819580, 1339.1713867
4: -393.8876648, 966.2492065, -329.2483215, 807.8594360, -1201.7470703, 1295.4975586

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015770
time: 0.81 seconds

## Relational analysis of IS_A2_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015770
time: 0.74 seconds

## BFS IS instance: IS_A2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -196.0597839, 868.2266846, -160.7262268, 714.6234131, -910.6832275, 1028.9528809
1: -245.0461578, 988.2661743, -201.2299500, 813.6099243, -1058.6558838, 1189.4960938
2: -249.9981689, 978.7294922, -205.7579346, 805.9060669, -1055.9041748, 1184.4874268
3: -396.1403503, 1027.0897217, -325.3089294, 845.2531738, -1241.3935547, 1352.3986816
4: -399.3693237, 979.3389893, -328.6820374, 806.4335938, -1205.8029785, 1308.0209961

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3020955
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020955
time: 0.83 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -164.6814880, 729.0076294, -924.9586792, 1033.0612793
1: -244.9847565, 988.4234009, -205.9361420, 829.8331299, -1074.8178711, 1194.3592529
2: -249.9962616, 978.9415283, -210.1320648, 821.9001465, -1071.8963623, 1189.0736084
3: -396.1255798, 1027.2198486, -332.7737427, 862.5795288, -1258.7049561, 1359.9931641
4: -399.4131775, 979.5528564, -335.4803162, 822.6964111, -1222.1096191, 1315.0332031

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007856, upper bound: 853.3007703
time: 0.81 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016270, upper bound: 853.3016270
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -196.0472412, 868.7943115, -1064.7454834, 1064.4270020
1: -244.9847565, 988.4234009, -245.1055756, 988.8947754, -1233.8795166, 1233.5281982
2: -249.9962616, 978.9415283, -250.1186981, 979.4112549, -1229.4073486, 1229.0601807
3: -396.1255798, 1027.2198486, -396.3183899, 1027.7099609, -1423.8355713, 1423.5382080
4: -399.4131775, 979.5528564, -399.6038513, 980.0239868, -1379.4371338, 1379.1567383

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019936, upper bound: 853.3018768
time: 0.78 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019684, upper bound: 853.3019773
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.34 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2975329, upper bound: 853.2996478
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2975010, upper bound: 853.2996478
IS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2968235, upper bound: 853.2985340
IS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2965499, upper bound: 853.2984998
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2917703, upper bound: 853.2844389
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2979167, upper bound: 853.2976750
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2952556, upper bound: 853.2951503
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2957008, upper bound: 853.2963442
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3011972, upper bound: 853.3011264
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3011972, upper bound: 853.3011264
IS_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3011654, upper bound: 853.3011264
IS_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3011654, upper bound: 853.3011264
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2920320, upper bound: 853.2844396
IS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2979705, upper bound: 853.2973437
IS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2958054, upper bound: 853.2975888
IS_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2946540, upper bound: 853.2946540
IS_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2958054, upper bound: 853.2990675
IS_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2946540, upper bound: 853.2961326
IS_A1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2983964, upper bound: 853.2982241
IS_A1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2986983, upper bound: 853.2983928
IS_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2961687, upper bound: 853.2984461
IS_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2984374, upper bound: 853.2996665
IS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3010395
IS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2997969, upper bound: 853.2997969
IS_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3008532, upper bound: 853.3010392
IS_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2997969, upper bound: 853.2997969
IS_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3003186, upper bound: 853.3007118
IS_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3019674, upper bound: 853.3018264
IS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
IS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3020955, upper bound: 853.3020276
IS_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2977580, upper bound: 853.2987189
IS_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2966066, upper bound: 853.2957841
IS_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2976023, upper bound: 853.2861533
IS_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3007442, upper bound: 853.3003009
IS_A2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2996070, upper bound: 853.2992113
IS_A2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3007442, upper bound: 853.3001090
IS_A2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3007849, upper bound: 853.3000499
IS_A2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3006694, upper bound: 853.3000548
IS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2991415, upper bound: 853.3006910
IS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2980852, upper bound: 853.2994484
IS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3021208, upper bound: 853.3024080
IS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3024080
IS_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2982241, upper bound: 853.2983964
IS_A2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2983928, upper bound: 853.2986984
IS_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2972080, upper bound: 853.2989454
IS_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2983048, upper bound: 853.3005978
IS_A2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2982556
IS_A2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2982695
IS_A2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2997722
IS_A2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2997861
IS_A2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015770
IS_A2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015770
IS_A2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3020955
IS_A2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020955
IS_A2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3007856, upper bound: 853.3007703
IS_A2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3016270, upper bound: 853.3016270
IS_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3019936, upper bound: 853.3018768
IS_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.34
Output dim: 0, lower bound: -853.3019684, upper bound: 853.3019773

## BFS IS instance: IS_A1_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -128.8350372, 574.2913818, -722.5134277, 784.0455933
1: -185.5556030, 746.0098877, -161.2933197, 653.5964355, -839.1520386, 907.3032227
2: -189.6679535, 739.3981323, -165.0812378, 647.5028076, -837.1707764, 904.4793701
3: -299.1638184, 775.3083496, -260.8292542, 678.6773071, -977.8411255, 1036.1375732
4: -302.1930847, 740.4500122, -263.1418457, 648.1230469, -950.3161011, 1003.5917969

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975167, upper bound: 853.2996478
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960129, upper bound: 853.2990312
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2955181, upper bound: 853.2986690
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -148.0500183, 654.4420166, -138.3276825, 614.3864136, -762.4363403, 792.7696533
1: -185.3428650, 745.1341553, -173.1211090, 699.2821655, -884.6249390, 918.2552490
2: -189.4498138, 738.5339355, -177.0674438, 692.7832031, -882.2330322, 915.6012573
3: -298.8182068, 774.3992920, -279.7284546, 726.5140991, -1025.3322754, 1054.1276855
4: -301.8421936, 739.5947876, -282.4146423, 693.4832153, -995.3254395, 1022.0093384

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975010, upper bound: 853.2996478
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2959855, upper bound: 853.2990329
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2954906, upper bound: 853.2986708
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -148.2220306, 655.2105713, -145.3942413, 646.4320679, -794.6541138, 800.6046753
1: -185.5556030, 746.0098877, -182.0883636, 735.8641968, -921.4197388, 928.0982666
2: -189.6679535, 739.3981323, -186.3109589, 729.1959839, -918.8639526, 925.7091064
3: -299.1638184, 775.3083496, -294.1707764, 764.4715576, -1063.6353760, 1069.4791260
4: -302.1930847, 740.4500122, -296.9245300, 730.2315063, -1032.4243164, 1037.3745117

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2940925, upper bound: 853.2967235
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2940925, upper bound: 853.2968967
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -148.0500183, 654.4420166, -152.8448486, 677.6488037, -825.6987305, 807.2868652
1: -185.3428650, 745.1341553, -191.3385162, 771.4152222, -956.7579956, 936.4726562
2: -189.4498138, 738.5339355, -195.6974030, 764.3942871, -953.8441162, 934.2313232
3: -298.8182068, 774.3992920, -308.8927612, 801.6753540, -1100.4935303, 1083.2919922
4: -301.8421936, 739.5947876, -312.0253601, 765.4360962, -1067.2779541, 1051.6201172

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2940515, upper bound: 853.2967025
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2940261, upper bound: 853.2968505
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -127.0278702, 563.1144409, -162.2422028, 717.7668457, -844.7947388, 725.3566284
1: -158.9505157, 641.1395264, -202.8611908, 817.0556641, -976.0061646, 844.0006714
2: -162.5271606, 635.0861816, -207.0121307, 809.2485962, -971.7756958, 842.0983276
3: -256.5744019, 666.0552979, -327.7374878, 849.3481445, -1105.9223633, 993.7927856
4: -259.2628784, 635.6389771, -330.4922485, 810.0374146, -1069.2998047, 966.1312256

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979167, upper bound: 853.2976750
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979167, upper bound: 853.2976750
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2979029, upper bound: 853.2968475
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961331, upper bound: 853.2968495
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2932751, upper bound: 853.2929727
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -146.0648804, 645.6864014, -160.3108215, 709.5798950, -855.6447754, 805.9971313
1: -182.8547516, 735.1871948, -200.4193268, 807.8021240, -990.6568604, 935.6065063
2: -186.9111023, 728.6476440, -204.5284424, 799.9448242, -986.8557129, 933.1760864
3: -294.7907715, 764.0739136, -323.8647766, 839.6726074, -1134.4633789, 1087.9387207
4: -297.8164062, 729.6941528, -326.6532898, 800.6696777, -1098.4858398, 1056.3472900

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2952556, upper bound: 853.2951503
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2952534, upper bound: 853.2946576
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2919902, upper bound: 853.2924398
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2950735, upper bound: 853.2949667
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2938792, upper bound: 853.2946188
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2952556, upper bound: 853.2951449
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2952556, upper bound: 853.2951449
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -146.0648804, 645.6864014, -160.0119019, 709.5555420, -855.6204224, 805.6982422
1: -182.8547516, 735.1871948, -200.0671539, 807.6859131, -990.5406494, 935.2543335
2: -186.9111023, 728.6476440, -204.2142487, 799.8270874, -986.7381592, 932.8618774
3: -294.7907715, 764.0739136, -323.5531921, 839.4380493, -1134.2287598, 1087.6270752
4: -297.8164062, 729.6941528, -326.2160339, 800.5352783, -1098.3515625, 1055.9100342

Time for backsubstitution: 1.97 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3024515
time: 0.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021378, upper bound: 853.3021378
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3024515
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -853.3021378, upper bound: 853.3021378

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -172.6161652, 763.3468018, -182.7681427, 808.7180176, -981.3341064, 946.1149292
1: -215.8617249, 868.9192505, -228.5348969, 920.6278076, -1136.4893799, 1097.4539795
2: -220.2556305, 860.7608643, -233.1255188, 911.7817383, -1132.0373535, 1093.8863525
3: -348.6884155, 903.1724243, -369.3415222, 956.9161377, -1305.6043701, 1272.5139160
4: -351.4360046, 861.5000000, -372.4694519, 912.2335815, -1263.6694336, 1233.9692383

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
time: 0.75 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
time: 0.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -203.9094696, 903.2384644, -182.0342102, 805.5654907, -1009.4749146, 1085.2727051
1: -254.9292297, 1028.1209717, -227.6165771, 917.0407104, -1171.9699707, 1255.7375488
2: -260.1297302, 1018.3677979, -232.1985016, 908.2293091, -1168.3588867, 1250.5662842
3: -412.1533813, 1068.4002686, -367.8658752, 953.1840210, -1365.3374023, 1436.2661133
4: -415.5236206, 1018.7225342, -370.9976807, 908.6749268, -1324.1984863, 1389.7202148

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3007123
time: 0.72 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020952
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.42 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3007123
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020952

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -181.7130585, 804.1934204, -955.9542847, 855.9424438
1: -190.0204926, 767.5302734, -227.2232971, 915.4792480, -1105.4997559, 994.7535400
2: -194.3727112, 760.5083618, -231.8014069, 906.6630859, -1101.0357666, 992.3097534
3: -306.9681091, 797.3639526, -367.2392273, 951.5597534, -1258.5278320, 1164.6031494
4: -309.9315796, 761.3359985, -370.3706970, 907.1113281, -1217.0428467, 1131.7064209

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.69 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -181.6816559, 804.0122681, -968.6937256, 910.6892700
1: -205.9361420, 829.8331299, -227.1777039, 915.2708130, -1121.2069092, 1057.0108643
2: -210.1320648, 821.9001465, -231.7403717, 906.4563599, -1116.5883789, 1053.6403809
3: -332.7737427, 862.5795288, -367.1641846, 951.3548584, -1284.1284180, 1229.7436523
4: -335.4803162, 822.6964111, -370.2813110, 906.9237671, -1242.4039307, 1192.9777832

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
time: 0.75 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -197.8976135, 871.8310547, -181.4418945, 802.8433838, -1000.7409668, 1053.2728271
1: -247.3440247, 992.4859009, -226.8764191, 913.9420166, -1161.2860107, 1219.3621826
2: -252.2371979, 983.4658203, -231.4418793, 905.1749878, -1157.4119873, 1214.9077148
3: -399.0464478, 1031.4986572, -366.6489868, 949.9750366, -1349.0214844, 1398.1477051
4: -402.1404419, 983.9295654, -369.7678223, 905.6326904, -1307.7728271, 1353.6973877

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.73 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006390
time: 0.71 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -203.8071899, 902.7960815, -182.0342102, 805.5654907, -1009.3726807, 1084.8302002
1: -254.8004913, 1027.6179199, -227.6165771, 917.0407104, -1171.8411865, 1255.2343750
2: -259.9992065, 1017.8665771, -232.1985016, 908.2293091, -1168.2285156, 1250.0649414
3: -411.9476318, 1067.8769531, -367.8658752, 953.1840210, -1365.1315918, 1435.7427979
4: -415.3203125, 1018.2191772, -370.9976807, 908.6749268, -1323.9952393, 1389.2167969

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 0.78 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.54 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3022252, upper bound: 853.3023840
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006390
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.54
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -150.5078430, 668.4323730, -168.2443848, 740.4166870, -890.9244385, 836.6767578
1: -188.4600983, 760.9351807, -210.3257904, 842.9163208, -1031.3764648, 971.2609863
2: -192.7745667, 754.0164185, -214.5335388, 835.3629761, -1028.1375732, 968.5498047
3: -304.3941040, 790.5433960, -339.0354309, 876.4241943, -1180.8179932, 1129.5788574
4: -307.3167114, 754.8905640, -341.9064026, 836.1650391, -1143.4816895, 1096.7969971

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -181.6117401, 803.7498169, -955.5107422, 855.8413086
1: -190.0204926, 767.5302734, -227.0953522, 914.9757080, -1104.9962158, 994.6256104
2: -194.3727112, 760.5083618, -231.6713257, 906.1613159, -1100.5340576, 992.1796875
3: -306.9681091, 797.3639526, -367.0347290, 951.0360718, -1258.0041504, 1164.3986816
4: -309.9315796, 761.3359985, -370.1683350, 906.6060791, -1216.5374756, 1131.5042725

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -161.3719482, 717.4226074, -882.1040649, 890.3795166
1: -205.9361420, 829.8331299, -202.0404663, 816.7867432, -1022.7229004, 1031.8734131
2: -210.1320648, 821.9001465, -206.5800781, 809.0697021, -1019.2017822, 1028.4802246
3: -332.7737427, 862.5795288, -326.6069031, 848.5523071, -1181.3256836, 1189.1860352
4: -335.4803162, 822.6964111, -329.9661560, 809.6101074, -1145.0903320, 1152.6625977

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -174.8621674, 774.4973755, -939.1788330, 903.8698120
1: -205.9361420, 829.8331299, -218.6441650, 881.6770020, -1087.6131592, 1048.4772949
2: -210.1320648, 821.9001465, -223.0339050, 873.0563965, -1083.1884766, 1044.9337158
3: -332.7737427, 862.5795288, -353.4781799, 916.4602661, -1249.2338867, 1216.0574951
4: -335.4803162, 822.6964111, -356.5664062, 873.5581665, -1209.0380859, 1179.2628174

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -180.3824768, 798.2865601, -973.8750000, 956.8692627
1: -219.7106323, 883.9606934, -225.5588684, 908.7568970, -1128.4675293, 1109.5191650
2: -224.4729156, 876.0089111, -230.1113739, 900.0214233, -1124.4943848, 1106.1202393
3: -354.5040283, 918.4062500, -364.5351868, 944.5815430, -1299.0855713, 1282.9414062
4: -357.8605652, 876.7702637, -367.6577148, 900.4758911, -1258.3363037, 1244.4279785

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.99 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -180.3327789, 798.0409546, -988.4614868, 1019.6137695
1: -238.0125122, 955.4159546, -225.4909668, 908.4749756, -1146.4873047, 1180.9067383
2: -242.7419586, 946.6467285, -230.0280151, 899.7404785, -1142.4824219, 1176.6745605
3: -384.0323486, 993.0491943, -364.4270630, 944.2992554, -1328.3315430, 1357.4763184
4: -387.0486145, 947.3689575, -367.5344849, 900.2136230, -1287.2620850, 1314.9034424

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006390
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006390
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -180.1696320, 801.5511475, -180.9931335, 801.0960083, -981.2655029, 982.5443115
1: -225.4840698, 912.3982544, -226.3222504, 911.9552002, -1137.4392090, 1138.7202148
2: -230.5822754, 903.7421265, -230.8916473, 903.1736450, -1133.7558594, 1134.6336670
3: -364.6455078, 947.7800293, -365.7905884, 947.8932495, -1312.5388184, 1313.5705566
4: -368.4310913, 904.3006592, -368.9262085, 903.6160889, -1272.0468750, 1273.2268066

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 0.70 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -180.9370422, 800.8165283, -996.7675781, 1049.3167725
1: -244.9847565, 988.4234009, -226.2462463, 911.6347656, -1156.6193848, 1214.6691895
2: -249.9962616, 978.9415283, -230.7999878, 902.8547974, -1152.8509521, 1209.7414551
3: -396.1255798, 1027.2198486, -365.6684570, 947.5712891, -1343.6968994, 1392.8881836
4: -399.4131775, 979.5528564, -368.7891846, 903.3162842, -1302.7294922, 1348.3420410

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.77 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.61 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001637
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006390
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3006390
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -150.4852905, 668.3262329, -148.7031708, 657.2008057, -807.6860962, 817.0294189
1: -188.4320374, 760.8145142, -186.1483459, 748.2800903, -936.7120361, 946.9627686
2: -192.7457886, 753.8978271, -190.2608185, 741.6431274, -934.3889160, 944.1586304
3: -304.3474731, 790.4183960, -300.1105957, 777.6807251, -1082.0281982, 1090.5290527
4: -307.2690735, 754.7731934, -303.1169128, 742.7159424, -1049.9849854, 1057.8901367

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -150.4906616, 668.3491211, -161.5639496, 711.8400269, -862.3306885, 829.9130859
1: -188.4388123, 760.8405762, -201.9722748, 810.4206543, -998.8594971, 962.8128052
2: -192.7526093, 753.9237671, -206.0324402, 802.9714966, -995.7240601, 959.9561768
3: -304.3581543, 790.4456177, -325.6631470, 842.6292114, -1146.9873047, 1116.1087646
4: -307.2798157, 754.7991333, -328.5767822, 803.7938232, -1111.0736084, 1083.3759766

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996777, upper bound: 853.2994385
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -171.4539490, 758.3616333, -910.1224976, 845.6834717
1: -190.0204926, 767.5302734, -214.4150391, 863.2473145, -1053.2678223, 981.9453125
2: -194.3727112, 760.5083618, -218.7951050, 855.1186523, -1049.4913330, 979.3033447
3: -306.9681091, 797.3639526, -346.3705444, 897.2719727, -1204.2401123, 1143.7344971
4: -309.9315796, 761.3359985, -349.1234436, 855.8538818, -1165.7854004, 1110.4594727

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -202.6755219, 897.8866577, -1049.6475830, 876.9050293
1: -190.0204926, 767.5302734, -253.3943481, 1022.0263672, -1212.0468750, 1020.9246216
2: -194.3727112, 760.5083618, -258.5769958, 1012.3239136, -1206.6965332, 1019.0853271
3: -306.9681091, 797.3639526, -409.6860962, 1062.0723877, -1369.0405273, 1207.0499268
4: -309.9315796, 761.3359985, -413.0531311, 1012.6976318, -1322.6291504, 1174.3890381

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -163.4345551, 723.1777344, -148.2220306, 655.2105713, -818.6450806, 871.3996582
1: -204.3810272, 823.1984863, -185.5556030, 746.0098877, -950.3908691, 1008.7540894
2: -208.5389709, 815.3747559, -189.6679535, 739.3981323, -947.9371338, 1005.0426636
3: -330.1931458, 855.7177734, -299.1638184, 775.3083496, -1105.5014648, 1154.8815918
4: -332.8596191, 816.2174683, -302.1930847, 740.4500122, -1073.3095703, 1118.4104004

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007997, upper bound: 853.3004490
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007997, upper bound: 853.3004184
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -161.2957306, 717.0905762, -881.7720337, 890.3033447
1: -205.9361420, 829.8331299, -201.9450073, 816.4095459, -1022.3457031, 1031.7780762
2: -210.1320648, 821.9001465, -206.4831696, 808.6948853, -1018.8269653, 1028.3831787
3: -332.7737427, 862.5795288, -326.4538574, 848.1599121, -1180.9334717, 1189.0332031
4: -335.4803162, 822.6964111, -329.8136902, 809.2348022, -1144.7149658, 1152.5101318

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3024502
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3024502
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -163.4397278, 723.2001953, -161.5639496, 711.8400269, -875.2797852, 884.7641602
1: -204.3875885, 823.2241211, -201.9722748, 810.4206543, -1014.8082275, 1025.1964111
2: -208.5456238, 815.4000854, -206.0324402, 802.9714966, -1011.5170898, 1021.4324951
3: -330.2036133, 855.7443237, -325.6631470, 842.6292114, -1172.8327637, 1181.4074707
4: -332.8701172, 816.2429810, -328.5767822, 803.7938232, -1136.6639404, 1144.8195801

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -174.7723389, 774.1043701, -938.7858276, 903.7799683
1: -205.9361420, 829.8331299, -218.5308075, 881.2310181, -1087.1671143, 1048.3638916
2: -210.1320648, 821.9001465, -222.9185486, 872.6120605, -1082.7441406, 1044.8186035
3: -332.7737427, 862.5795288, -353.2970886, 915.9963989, -1248.7696533, 1215.8763428
4: -335.4803162, 822.6964111, -356.3869629, 873.1104126, -1208.5905762, 1179.0833740

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -170.9933167, 756.2204590, -931.8089600, 947.4801025
1: -219.7106323, 883.9606934, -213.8410034, 860.8100586, -1080.5205078, 1097.8013916
2: -224.4729156, 876.0089111, -218.2080383, 852.7203369, -1077.1932373, 1094.2169189
3: -354.5040283, 918.4062500, -345.4206543, 894.7496338, -1249.2536621, 1263.8269043
4: -357.8605652, 876.7702637, -348.1596375, 853.4696655, -1211.3302002, 1224.9299316

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.74 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -201.6742554, 893.1043091, -1068.6923828, 978.1610718
1: -219.7106323, 883.9606934, -252.1519470, 1016.5844727, -1236.2950439, 1136.1123047
2: -224.4729156, 876.0089111, -257.2937927, 1006.9655762, -1231.4384766, 1133.3027344
3: -354.5040283, 918.4062500, -407.6156921, 1056.4721680, -1410.9761963, 1326.0219727
4: -357.8605652, 876.7702637, -410.9565125, 1007.3972168, -1365.2574463, 1287.7268066

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -160.0081635, 711.4621582, -901.8826904, 999.2892456
1: -238.0125122, 955.4159546, -200.3355560, 810.0100098, -1048.0222168, 1155.7512207
2: -242.7419586, 946.6467285, -204.8480225, 802.3525391, -1045.0944824, 1151.4947510
3: -384.0323486, 993.0491943, -323.8621216, 841.5189209, -1225.5510254, 1316.9112549
4: -387.0486145, 947.3689575, -327.2227783, 802.8887939, -1189.9372559, 1274.5917969

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006390
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -173.4916840, 768.4409180, -958.8614502, 1012.7727051
1: -238.0125122, 955.4159546, -216.9299469, 874.7825317, -1112.7947998, 1172.3457031
2: -242.7419586, 946.6467285, -221.2929688, 866.2412720, -1108.9832764, 1167.9396973
3: -384.0323486, 993.0491943, -350.6987000, 909.3026123, -1293.3349609, 1343.7479248
4: -387.0486145, 947.3689575, -353.7789001, 866.7467041, -1253.7952881, 1301.1477051

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006390
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -180.1696320, 801.5511475, -171.5556488, 758.8069458, -938.9764404, 973.1067505
1: -225.4840698, 912.3982544, -214.5434570, 863.7532349, -1089.2373047, 1126.9414062
2: -230.5822754, 903.7421265, -218.9257812, 855.6226807, -1086.2049561, 1122.6677246
3: -364.6455078, 947.7800293, -346.5758972, 897.7979736, -1262.4433594, 1294.3555908
4: -368.4310913, 904.3006592, -349.3266602, 856.3610840, -1224.7921143, 1253.6273193

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 0.80 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 0.74 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -180.1696320, 801.5511475, -202.7772675, 898.3272705, -1078.4968262, 1004.3283691
1: -225.4840698, 912.3982544, -253.5224915, 1022.5271606, -1248.0112305, 1165.9202881
2: -230.5822754, 903.7421265, -258.7069092, 1012.8228149, -1243.4050293, 1162.4489746
3: -364.6455078, 947.7800293, -409.8908386, 1062.5931396, -1427.2386475, 1357.6707764
4: -368.4310913, 904.3006592, -413.2554016, 1013.1986084, -1381.6296387, 1317.5560303

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 1.24 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -160.5381317, 713.9136963, -909.8647461, 1028.9178467
1: -244.9847565, 988.4234009, -200.9977264, 812.8023071, -1057.7871094, 1189.4207764
2: -249.9962616, 978.9415283, -205.5271301, 805.1016235, -1055.0979004, 1184.4686279
3: -396.1255798, 1027.2198486, -324.9513550, 844.4077148, -1240.5333252, 1352.1708984
4: -399.4131775, 979.5528564, -328.3266907, 805.6235352, -1205.0367432, 1307.8795166

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3020276
time: 0.87 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.88 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -174.0997467, 771.2259521, -967.1770020, 1042.4794922
1: -244.9847565, 988.4234009, -217.6902161, 877.9528198, -1122.9373779, 1206.1134033
2: -249.9962616, 978.9415283, -222.0699921, 869.3679810, -1119.3642578, 1201.0114746
3: -396.1255798, 1027.2198486, -351.9469604, 912.5867920, -1308.7122803, 1379.1666260
4: -399.4131775, 979.5528564, -355.0382080, 869.8622437, -1269.2753906, 1334.5910645

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3020276
time: 0.81 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.80 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007997, upper bound: 853.3004490
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007997, upper bound: 853.3004184
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3024502
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3024502
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001527
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023826
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3001050, upper bound: 853.3006443
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006390
IS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.2984205, upper bound: 853.2984205
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.2984205, upper bound: 853.3006390
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3020952, upper bound: 853.3020276
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3020276
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.2998086, upper bound: 853.3020276
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.80
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020276

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -147.5671082, 655.8410034, -148.7031708, 657.2008057, -804.7678833, 804.5441895
1: -184.8032990, 746.5563965, -186.1483459, 748.2800903, -933.0833740, 932.7046509
2: -189.0801392, 739.8526001, -190.2608185, 741.6431274, -930.7232666, 930.1134033
3: -298.5374756, 775.5886230, -300.1105957, 777.6807251, -1076.2182617, 1075.6989746
4: -301.2781982, 740.8801880, -303.1169128, 742.7159424, -1043.9941406, 1043.9969482

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996298, upper bound: 853.2975329
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985340, upper bound: 853.2968235
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -154.8359222, 686.2690430, -148.0820465, 654.4680786, -809.3039551, 834.3510742
1: -193.8242340, 781.2233887, -185.3808441, 745.1654053, -938.9896240, 966.6041870
2: -198.2376099, 774.1497803, -189.4780731, 738.5686646, -936.8062744, 963.6277466
3: -312.8889465, 811.8740234, -298.8674927, 774.4434814, -1087.3322754, 1110.7414551
4: -316.0261536, 775.1775513, -301.8652344, 739.6610718, -1055.6872559, 1077.0427246

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996340, upper bound: 853.2975001
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984998, upper bound: 853.2965499
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -147.5719452, 655.8615723, -161.5639496, 711.8400269, -859.4119263, 817.4255371
1: -184.8093872, 746.5800171, -201.9722748, 810.4206543, -995.2300415, 948.5523071
2: -189.0863495, 739.8761597, -206.0324402, 802.9714966, -992.0578613, 945.9085693
3: -298.5470276, 775.6130981, -325.6631470, 842.6292114, -1141.1762695, 1101.2762451
4: -301.2878418, 740.9037476, -328.5767822, 803.7938232, -1105.0816650, 1069.4804688

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996777, upper bound: 853.2994385
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003989, upper bound: 853.2998896
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001256
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001365
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -154.8408508, 686.2907715, -160.9820404, 709.2457886, -864.0865479, 847.2728271
1: -193.8304596, 781.2479248, -201.2501373, 807.4667358, -1001.2971802, 982.4979248
2: -198.2439270, 774.1744995, -205.2929382, 800.0435181, -998.2873535, 979.4674072
3: -312.8988342, 811.8994141, -324.4940491, 839.5659180, -1152.4647217, 1136.3933105
4: -316.0360413, 775.2023315, -327.4000244, 800.8914185, -1116.9274902, 1102.6022949

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996441, upper bound: 853.2991691
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004603, upper bound: 853.2998099
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001087
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -151.6844177, 673.8961792, -825.6569824, 825.9139404
1: -190.0204926, 767.5302734, -189.9245453, 767.1513672, -957.1718750, 957.4547729
2: -194.3727112, 760.5083618, -194.2752380, 760.1318359, -954.5045166, 954.7835693
3: -306.9681091, 797.3639526, -306.8143005, 796.9697876, -1103.9378662, 1104.1782227
4: -309.9315796, 761.3359985, -309.7783203, 760.9587402, -1070.8902588, 1071.1141357

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025376, upper bound: 853.3025082
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025265, upper bound: 853.3025265
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -164.5898590, 728.6068726, -880.3677368, 838.8193970
1: -190.0204926, 767.5302734, -205.8206482, 829.3784790, -1019.3989868, 973.3508301
2: -194.3727112, 760.5083618, -210.0145264, 821.4472046, -1015.8199463, 970.5228271
3: -306.9681091, 797.3639526, -332.5891418, 862.1064453, -1169.0745850, 1129.9530029
4: -309.9315796, 761.3359985, -335.2973022, 822.2402344, -1132.1718750, 1096.6333008

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025267, upper bound: 853.3024948
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024948, upper bound: 853.3024948
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -180.1680603, 801.5437622, -953.3046265, 854.3975830
1: -190.0204926, 767.5302734, -225.4821167, 912.3897705, -1102.4102783, 993.0123901
2: -194.3727112, 760.5083618, -230.5803070, 903.7339478, -1098.1064453, 991.0886841
3: -306.9681091, 797.3639526, -364.6423035, 947.7713623, -1254.7395020, 1162.0062256
4: -309.9315796, 761.3359985, -368.4276733, 904.2925415, -1214.2241211, 1129.7635498

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010657, upper bound: 853.3016036
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 1.06 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -195.9478760, 868.3646851, -1020.1254883, 870.1773682
1: -190.0204926, 767.5302734, -244.9806824, 988.4062500, -1178.4265137, 1012.5109863
2: -194.3727112, 760.5083618, -249.9921112, 978.9247437, -1173.2973633, 1010.5003662
3: -306.9681091, 797.3639526, -396.1189270, 1027.2017822, -1334.1699219, 1193.4829102
4: -309.9315796, 761.3359985, -399.4063721, 979.5361328, -1289.4677734, 1160.7423096

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010657, upper bound: 853.3016036
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 1.02 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 1.19 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -160.2605438, 709.5877075, -148.2220306, 655.2105713, -815.4710693, 857.8096313
1: -200.4398651, 807.6890869, -185.5556030, 746.0098877, -946.4497070, 993.2445679
2: -204.5495300, 800.0691528, -189.6679535, 739.3981323, -943.9476318, 989.7371216
3: -323.8881226, 839.5923462, -299.1638184, 775.3083496, -1099.1964111, 1138.7561035
4: -326.3717346, 801.0668335, -302.1930847, 740.4500122, -1066.8215332, 1103.2597656

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977444, upper bound: 853.2987189
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2965544, upper bound: 853.2957841
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -167.8661499, 742.2500000, -147.7365265, 653.0407104, -820.9066772, 889.9865112
1: -209.8907013, 844.8943481, -184.9548645, 743.5371094, -953.4277954, 1029.8489990
2: -214.2031250, 836.8511353, -189.0521088, 736.9585571, -951.1616821, 1025.9030762
3: -338.9692078, 878.3427124, -298.1876526, 772.7414551, -1111.7106934, 1176.5303955
4: -341.7855835, 837.8731689, -301.2019348, 738.0353394, -1079.8208008, 1139.0750732

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995568, upper bound: 853.2975001
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004426, upper bound: 853.2993243
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003947, upper bound: 853.2993525
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -151.6844177, 673.8961792, -838.5775757, 880.6920166
1: -205.9361420, 829.8331299, -189.9245453, 767.1513672, -973.0875244, 1019.7576904
2: -210.1320648, 821.9001465, -194.2752380, 760.1318359, -970.2639160, 1016.1754150
3: -332.7737427, 862.5795288, -306.8143005, 796.9697876, -1129.7434082, 1169.3935547
4: -335.4803162, 822.6964111, -309.7783203, 760.9587402, -1096.4388428, 1132.4747314

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991296, upper bound: 853.3006625
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980852, upper bound: 853.2994484
time: 0.83 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -180.1696320, 801.5511475, -966.2325439, 909.1771851
1: -205.9361420, 829.8331299, -225.4840698, 912.3982544, -1118.3341064, 1055.3171387
2: -210.1320648, 821.9001465, -230.5822754, 903.7421265, -1113.8739014, 1052.4822998
3: -332.7737427, 862.5795288, -364.6455078, 947.7800293, -1280.5534668, 1227.2248535
4: -335.4803162, 822.6964111, -368.4310913, 904.3006592, -1239.7810059, 1191.1274414

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991296, upper bound: 853.3006625
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980852, upper bound: 853.2994484
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -160.2657013, 709.6100464, -161.5639496, 711.8400269, -872.1057129, 871.1740112
1: -200.4463959, 807.7144165, -201.9722748, 810.4206543, -1010.8670044, 1009.6867065
2: -204.5561676, 800.0944824, -206.0324402, 802.9714966, -1007.5276489, 1006.1269531
3: -323.8984070, 839.6187744, -325.6631470, 842.6292114, -1166.5275879, 1165.2819824
4: -326.3821716, 801.0921631, -328.5767822, 803.7938232, -1130.1759033, 1129.6689453

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977351, upper bound: 853.2984038
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995092, upper bound: 853.2986277
time: 1.14 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995092, upper bound: 853.2995429
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -167.8714294, 742.2732544, -160.9820404, 709.2457886, -877.1170654, 903.2553101
1: -209.8973694, 844.9208374, -201.2501373, 807.4667358, -1017.3641357, 1046.1708984
2: -214.2098999, 836.8775024, -205.2929382, 800.0435181, -1014.2534180, 1042.1704102
3: -338.9799805, 878.3701172, -324.4940491, 839.5659180, -1178.5458984, 1202.8640137
4: -341.7963257, 837.8997192, -327.4000244, 800.8914185, -1142.6876221, 1165.2995605

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996758, upper bound: 853.2993027
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004241, upper bound: 853.2995110
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -164.5898590, 728.6068726, -893.2883301, 893.5974731
1: -205.9361420, 829.8331299, -205.8206482, 829.3784790, -1035.3143311, 1035.6536865
2: -210.1320648, 821.9001465, -210.0145264, 821.4472046, -1031.5791016, 1031.9146729
3: -332.7737427, 862.5795288, -332.5891418, 862.1064453, -1194.8800049, 1195.1684570
4: -335.4803162, 822.6964111, -335.2973022, 822.2402344, -1157.7205811, 1157.9936523

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -195.9511719, 868.3797607, -1033.0612793, 924.9586792
1: -205.9361420, 829.8331299, -244.9847565, 988.4234009, -1194.3592529, 1074.8178711
2: -210.1320648, 821.9001465, -249.9962616, 978.9415283, -1189.0736084, 1071.8962402
3: -332.7737427, 862.5795288, -396.1255798, 1027.2198486, -1359.9931641, 1258.7049561
4: -335.4803162, 822.6964111, -399.4131775, 979.5528564, -1315.0332031, 1222.1096191

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991296, upper bound: 853.3005596
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
time: 1.17 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -151.1551361, 671.4735107, -847.0619507, 927.6419067
1: -219.7106323, 883.9606934, -189.2634125, 764.3970947, -984.1076660, 1073.2238770
2: -224.4729156, 876.0089111, -193.5974579, 757.4122314, -981.8850098, 1069.6063232
3: -354.5040283, 918.4062500, -305.7319946, 794.1188354, -1148.6228027, 1224.1381836
4: -357.8605652, 876.7702637, -308.6838989, 758.2476196, -1116.1079102, 1185.4541016

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977444
time: 0.77 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2965544
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -164.1051941, 726.3739014, -901.9623413, 940.5919800
1: -219.7106323, 883.9606934, -205.2160034, 826.8354492, -1046.5461426, 1089.1767578
2: -224.4729156, 876.0089111, -209.3968201, 818.9435425, -1043.4165039, 1085.4057617
3: -354.5040283, 918.4062500, -331.5913391, 859.4741211, -1213.9781494, 1249.9974365
4: -357.8605652, 876.7702637, -334.2881165, 819.7504883, -1177.6110840, 1211.0583496

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977444
time: 1.08 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2965544
time: 0.71 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -179.1887970, 796.9689941, -972.5574341, 955.6755981
1: -219.7106323, 883.9606934, -224.2534180, 907.1846313, -1126.8952637, 1108.2139893
2: -224.4729156, 876.0089111, -229.3383484, 898.6039429, -1123.0769043, 1105.3471680
3: -354.5040283, 918.4062500, -362.6120605, 942.3944702, -1296.8984375, 1281.0183105
4: -357.8605652, 876.7702637, -366.4065857, 899.1936646, -1257.0541992, 1243.1768799

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987087, upper bound: 853.2984227
time: 0.76 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987087, upper bound: 853.3006443
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -175.5885162, 776.4868164, -194.8589630, 863.2171631, -1038.8054199, 971.3457642
1: -219.7106323, 883.9606934, -243.6256104, 982.5526733, -1202.2633057, 1127.5860596
2: -224.4729156, 876.0089111, -248.5954132, 973.1543579, -1197.6273193, 1124.6041260
3: -354.5040283, 918.4062500, -393.8732910, 1021.1741333, -1375.6782227, 1312.2795410
4: -357.8605652, 876.7702637, -397.1448669, 973.8043213, -1331.6647949, 1273.9151611

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987087, upper bound: 853.2984227
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987087, upper bound: 853.3006443
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -189.9394073, 837.0470581, -147.5285950, 651.9910889, -841.9304810, 984.5756226
1: -237.4156952, 952.8815308, -184.6810608, 742.3428345, -979.7584229, 1137.5623779
2: -242.1350250, 944.1407471, -188.7850342, 735.7752075, -977.9102173, 1132.9257812
3: -383.0432129, 990.4287109, -297.7403870, 771.5152588, -1154.5584717, 1288.1688232
4: -386.0486755, 944.8957520, -300.8055725, 736.8327637, -1122.8814697, 1245.7010498

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2971606, upper bound: 853.2957994
time: 0.73 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975759, upper bound: 853.2971857
time: 0.73 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977842, upper bound: 853.2979181
time: 1.11 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -160.1644745, 712.2080688, -902.6286011, 999.4454956
1: -238.0125122, 955.4159546, -200.5304565, 810.8622437, -1048.8745117, 1155.9462891
2: -242.7419586, 946.6467285, -205.0494843, 803.1867676, -1045.9287109, 1151.6959229
3: -384.0323486, 993.0491943, -324.1866455, 842.3988037, -1226.4311523, 1317.2358398
4: -387.0486145, 947.3689575, -327.5538025, 803.7168579, -1190.7652588, 1274.9226074

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2971606, upper bound: 853.2994623
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983315, upper bound: 853.3007070
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984257, upper bound: 853.3007070
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -189.9490509, 837.0900269, -160.6460571, 707.6363525, -897.5853882, 997.7359619
1: -237.4274750, 952.9304199, -200.8259277, 805.6322632, -1043.0596924, 1153.7563477
2: -242.1471863, 944.1889038, -204.8688049, 798.2246094, -1040.3717041, 1149.0577393
3: -383.0623474, 990.4796143, -323.8040466, 837.6757812, -1220.7381592, 1314.2835693
4: -386.0683289, 944.9437256, -326.7267761, 799.1188354, -1185.1871338, 1271.6705322

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2982556
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2982695
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -173.7252197, 769.5013428, -959.9218750, 1013.0062256
1: -238.0125122, 955.4159546, -217.2208252, 875.9909058, -1114.0032959, 1172.6365967
2: -242.7419586, 946.6467285, -221.5894165, 867.4327393, -1110.1746826, 1168.2360840
3: -384.0323486, 993.0491943, -351.1773682, 910.5562744, -1294.5886230, 1344.2265625
4: -387.0486145, 947.3689575, -354.2593079, 867.9348145, -1254.9832764, 1301.6279297

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2997717
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2997860
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -180.1680603, 801.5437622, -151.7609406, 674.2295532, -854.3975830, 953.3046265
1: -225.4821167, 912.3897705, -190.0204926, 767.5302734, -993.0123901, 1102.4102783
2: -230.5803070, 903.7339478, -194.3727112, 760.5083618, -991.0886230, 1098.1064453
3: -364.6423035, 947.7713623, -306.9681091, 797.3639526, -1162.0062256, 1254.7395020
4: -368.4276733, 904.2925415, -309.9315796, 761.3359985, -1129.7635498, 1214.2241211

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006625, upper bound: 853.2991296
time: 0.70 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -180.1696320, 801.5511475, -164.6814880, 729.0076294, -909.1771851, 966.2325439
1: -225.4840698, 912.3982544, -205.9361420, 829.8331299, -1055.3171387, 1118.3341064
2: -230.5822754, 903.7421265, -210.1320648, 821.9001465, -1052.4824219, 1113.8739014
3: -364.6455078, 947.7800293, -332.7737427, 862.5795288, -1227.2248535, 1280.5534668
4: -368.4310913, 904.3006592, -335.4803162, 822.6964111, -1191.1274414, 1239.7810059

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006625, upper bound: 853.2991296
time: 1.06 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -180.1696320, 801.5511475, -180.2634430, 801.9572144, -982.1267700, 981.8145752
1: -225.4840698, 912.3982544, -225.6020966, 912.8598022, -1138.3438721, 1138.0001221
2: -230.5822754, 903.7421265, -230.7018127, 904.2016602, -1134.7839355, 1134.4437256
3: -364.6455078, 947.7800293, -364.8340759, 948.2597046, -1312.9049072, 1312.6138916
4: -368.4310913, 904.3006592, -368.6170349, 904.7614136, -1273.1923828, 1272.9177246

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987600, upper bound: 853.2984167
time: 0.85 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987600, upper bound: 853.3020276
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -180.1696320, 801.5511475, -196.0472412, 868.7943115, -1048.9639893, 997.5983887
1: -225.4840698, 912.3982544, -245.1055756, 988.8947754, -1214.3789062, 1157.5032959
2: -230.5822754, 903.7421265, -250.1186981, 979.4112549, -1209.9931641, 1153.8607178
3: -364.6455078, 947.7800293, -396.3183899, 1027.7099609, -1392.3554688, 1344.0983887
4: -368.4310913, 904.3006592, -399.6038513, 980.0239868, -1348.4547119, 1303.9045410

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987600, upper bound: 853.2984167
time: 0.80 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2987600, upper bound: 853.3020276
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -195.9478760, 868.3646851, -151.7609406, 674.2295532, -870.1773682, 1020.1255493
1: -244.9806824, 988.4062500, -190.0204926, 767.5302734, -1012.5109863, 1178.4266357
2: -249.9921112, 978.9247437, -194.3727112, 760.5083618, -1010.5003662, 1173.2973633
3: -396.1189270, 1027.2017822, -306.9681091, 797.3639526, -1193.4829102, 1334.1699219
4: -399.4063721, 979.5361328, -309.9315796, 761.3359985, -1160.7421875, 1289.4677734

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3009216, upper bound: 853.3008501
time: 0.83 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015764
time: 0.78 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020952
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -180.2634430, 801.9572144, -997.9082642, 1048.6431885
1: -244.9847565, 988.4234009, -225.6020966, 912.8598022, -1157.8443604, 1214.0251465
2: -249.9962616, 978.9415283, -230.7018127, 904.2016602, -1154.1976318, 1209.6433105
3: -396.1255798, 1027.2198486, -364.8340759, 948.2597046, -1344.3851318, 1392.0535889
4: -399.4131775, 979.5528564, -368.6170349, 904.7614136, -1304.1745605, 1348.1699219

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015764
time: 0.79 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020952
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -164.6814880, 729.0076294, -924.9586792, 1033.0612793
1: -244.9847565, 988.4234009, -205.9361420, 829.8331299, -1074.8178711, 1194.3592529
2: -249.9962616, 978.9415283, -210.1320648, 821.9001465, -1071.8963623, 1189.0736084
3: -396.1255798, 1027.2198486, -332.7737427, 862.5795288, -1258.7049561, 1359.9931641
4: -399.4131775, 979.5528564, -335.4803162, 822.6964111, -1222.1096191, 1315.0332031

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007856, upper bound: 853.3007695
time: 0.79 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016270, upper bound: 853.3016270
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -195.9511719, 868.3797607, -196.0472412, 868.7943115, -1064.7454834, 1064.4270020
1: -244.9847565, 988.4234009, -245.1055756, 988.8947754, -1233.8795166, 1233.5281982
2: -249.9962616, 978.9415283, -250.1186981, 979.4112549, -1229.4073486, 1229.0601807
3: -396.1255798, 1027.2198486, -396.3183899, 1027.7099609, -1423.8355713, 1423.5382080
4: -399.4131775, 979.5528564, -399.6038513, 980.0239868, -1379.4371338, 1379.1567383

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019936, upper bound: 853.3018768
time: 0.72 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019684, upper bound: 853.3019773
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.82 seconds
IS_A1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2996298, upper bound: 853.2975329
IS_A1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2985340, upper bound: 853.2968235
IS_A1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2996340, upper bound: 853.2975001
IS_A1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2984998, upper bound: 853.2965499
IS_A1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001256
IS_A1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001365
IS_A1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001087
IS_A1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3007945, upper bound: 853.3001208
IS_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3025376, upper bound: 853.3025082
IS_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3025265, upper bound: 853.3025265
IS_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3025267, upper bound: 853.3024948
IS_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3024948, upper bound: 853.3024948
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2977444, upper bound: 853.2987189
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2965544, upper bound: 853.2957841
IS_A1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3004426, upper bound: 853.2993243
IS_A1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3003947, upper bound: 853.2993525
IS_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2991296, upper bound: 853.3006625
IS_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2980852, upper bound: 853.2994484
IS_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2991296, upper bound: 853.3006625
IS_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2980852, upper bound: 853.2994484
IS_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2995092, upper bound: 853.2986277
IS_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2995092, upper bound: 853.2995429
IS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2996758, upper bound: 853.2993027
IS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3004241, upper bound: 853.2995110
IS_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023716
IS_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023398
IS_A2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977444
IS_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2965544
IS_A2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987189, upper bound: 853.2977444
IS_A2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2957841, upper bound: 853.2965544
IS_A2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987087, upper bound: 853.2984227
IS_A2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987087, upper bound: 853.3006443
IS_A2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987087, upper bound: 853.2984227
IS_A2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987087, upper bound: 853.3006443
IS_A2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2975759, upper bound: 853.2971857
IS_A2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2977842, upper bound: 853.2979181
IS_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2983315, upper bound: 853.3007070
IS_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2984257, upper bound: 853.3007070
IS_A2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2982556
IS_A2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2982695
IS_A2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2982632, upper bound: 853.2997717
IS_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2982682, upper bound: 853.2997860
IS_A2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3006625, upper bound: 853.2991296
IS_A2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3006625, upper bound: 853.2991296
IS_A2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2994484, upper bound: 853.2980852
IS_A2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987600, upper bound: 853.2984167
IS_A2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987600, upper bound: 853.3020276
IS_A2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987600, upper bound: 853.2984167
IS_A2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.2987600, upper bound: 853.3020276
IS_A2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015764
IS_A2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020952
IS_A2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3013463, upper bound: 853.3015764
IS_A2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3020276, upper bound: 853.3020952
IS_A2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3007856, upper bound: 853.3007695
IS_A2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3016270, upper bound: 853.3016270
IS_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3019936, upper bound: 853.3018768
IS_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.82
Output dim: 0, lower bound: -853.3019684, upper bound: 853.3019773

## BFS IS instance: IS_A1_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -127.9733047, 570.3599854, -147.2204895, 650.7705078, -778.7436523, 717.5804443
1: -160.2199402, 649.1275635, -184.3025818, 740.9658203, -901.1857300, 833.4300537
2: -163.9870605, 643.0961914, -188.3765717, 734.3809204, -898.3679810, 831.4727783
3: -259.0629272, 674.0462646, -297.1372070, 770.0771484, -1029.1401367, 971.1834717
4: -261.3545227, 643.7449951, -300.1516418, 735.4373169, -996.7918091, 943.8966064

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985340, upper bound: 853.2968235
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985340, upper bound: 853.2968235
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -144.5752716, 642.6487427, -148.5628510, 656.5838013, -801.1589966, 791.2116089
1: -181.0689850, 731.5610352, -185.9727783, 747.5787354, -928.6477051, 917.5337524
2: -185.2665253, 724.9597168, -190.0815582, 740.9462891, -926.2128296, 915.0412598
3: -292.4904480, 760.0216064, -299.8266907, 776.9524536, -1069.4428711, 1059.8482666
4: -295.2177734, 726.0251465, -302.8328247, 742.0184937, -1037.2363281, 1028.8579102

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985340, upper bound: 853.2968235
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985340, upper bound: 853.2968235
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -137.2923126, 609.5884399, -146.6001129, 648.0436401, -785.3359375, 756.1884155
1: -171.8286438, 693.8373413, -183.5353394, 737.8576660, -909.6862183, 877.3726807
2: -175.7536621, 687.3948975, -187.5949554, 731.3132324, -907.0668945, 874.9898682
3: -277.5927429, 720.8975830, -295.8962097, 766.8458252, -1044.4385986, 1016.7938232
4: -280.2905579, 688.1221924, -298.9027710, 732.3886108, -1012.6791382, 987.0249634

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984998, upper bound: 853.2965499
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984998, upper bound: 853.2965499
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -151.9357300, 673.4328613, -147.9417572, 653.8496094, -805.7853394, 821.3745728
1: -190.2055969, 766.6301270, -185.2052765, 744.4624023, -934.6679688, 951.8353882
2: -194.5382538, 759.6633911, -189.2988129, 737.8703613, -932.4086304, 948.9621582
3: -307.0193787, 796.7252197, -298.5834045, 773.7136841, -1080.7327881, 1095.3085938
4: -310.1429443, 760.7270508, -301.5807495, 738.9623413, -1049.1052246, 1062.3078613

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984998, upper bound: 853.2965499
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984998, upper bound: 853.2965499
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -144.7674866, 643.8699951, -161.5639496, 711.8400269, -856.6074219, 805.4339600
1: -181.2846985, 732.9246826, -201.9722748, 810.4206543, -991.7053223, 934.8969116
2: -185.5113678, 726.3211670, -206.0324402, 802.9714966, -988.4828491, 932.3536377
3: -292.9255371, 761.3784790, -325.6631470, 842.6292114, -1135.5546875, 1087.0416260
4: -295.6362305, 727.2963867, -328.5767822, 803.7938232, -1099.4300537, 1055.8731689

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996777, upper bound: 853.2994385
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003989, upper bound: 853.2998815
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2980251, upper bound: 853.2986578
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999913, upper bound: 853.2997422
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999001, upper bound: 853.2993995
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007463, upper bound: 853.3000696
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -147.2778473, 654.3173218, -160.6028290, 707.6149292, -854.8927612, 814.9201660
1: -184.4165192, 744.8488770, -200.7690277, 805.6174927, -990.0339966, 945.6177979
2: -188.7064362, 738.1384888, -204.8156586, 798.2047729, -986.9111328, 942.9541016
3: -297.8546753, 773.8693848, -323.7165833, 837.6325073, -1135.4868164, 1097.5859375
4: -300.7111511, 739.2116089, -326.6531067, 799.0015259, -1099.7126465, 1065.8646240

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001364, upper bound: 853.2998088
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007793, upper bound: 853.2999980
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000515, upper bound: 853.2990214
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -151.8583527, 673.5072632, -160.9820404, 709.2457886, -861.1040649, 834.4891968
1: -190.0756836, 766.7003784, -201.2501373, 807.4667358, -997.5424194, 967.9503784
2: -194.4326477, 759.7142944, -205.2929382, 800.0435181, -994.4761353, 965.0072021
3: -306.9183960, 796.7451782, -324.4940491, 839.5659180, -1146.4842529, 1121.2388916
4: -310.0461426, 760.6693115, -327.4000244, 800.8914185, -1110.9375000, 1088.0693359

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996441, upper bound: 853.2991691
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004603, upper bound: 853.2998099
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999947, upper bound: 853.2996342
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2984508, upper bound: 853.2982904
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007090, upper bound: 853.2999801
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -158.0249939, 699.2992554, -160.0215454, 705.0410767, -863.0660400, 859.3206787
1: -197.7764435, 796.1027832, -200.0469208, 802.6865234, -1000.4628906, 996.1496582
2: -202.2156525, 788.8867188, -204.0769196, 795.2969360, -997.5125122, 992.9636230
3: -319.0797729, 827.4348145, -322.5517273, 834.5907593, -1153.6705322, 1149.9865723
4: -322.2159119, 790.0705566, -325.4812012, 796.1172485, -1118.3331299, 1115.5517578

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003264, upper bound: 853.2997528
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996758, upper bound: 853.2993027
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004241, upper bound: 853.2995110
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -141.5634308, 627.7958984, -779.5567627, 815.7929688
1: -190.0204926, 767.5302734, -177.2257843, 714.6244507, -904.6449585, 944.7559204
2: -194.3727112, 760.5083618, -181.4008636, 708.4213867, -902.7940674, 941.9092407
3: -306.9681091, 797.3639526, -285.9134827, 742.2609253, -1049.2290039, 1083.2773438
4: -309.9315796, 761.3359985, -288.3812256, 709.6760254, -1019.6075439, 1049.7170410

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025267, upper bound: 853.3024764
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023495, upper bound: 853.3021803
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -150.8117676, 669.9483643, -164.7132568, 730.7569580, -881.5687256, 834.6614990
1: -188.8344116, 762.6622314, -206.2045593, 831.8223267, -1020.6567383, 968.8668213
2: -193.1705017, 755.7137451, -210.8065796, 824.3630371, -1017.5335083, 966.5203247
3: -305.0177612, 792.3120117, -332.8459167, 864.3365479, -1169.3542480, 1125.1578369
4: -307.9742126, 756.5613403, -336.2207947, 825.3427124, -1133.3168945, 1092.7821045

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025249, upper bound: 853.3024948
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024948, upper bound: 853.3024948
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -161.4230499, 715.0408325, -866.8016968, 835.6525269
1: -190.0204926, 767.5302734, -201.8878174, 813.8959351, -1003.9164429, 969.4180908
2: -194.3727112, 760.5083618, -206.0331573, 806.1691895, -1000.5418701, 966.5414429
3: -306.9681091, 797.3639526, -326.2968445, 846.0095215, -1152.9776611, 1123.6607666
4: -309.9315796, 761.3359985, -328.8228149, 807.1191406, -1117.0507812, 1090.1588135

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007689, upper bound: 853.2994418
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996299, upper bound: 853.2985016
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -150.6461945, 669.2319336, -169.1817017, 748.5742798, -899.2204590, 838.4135132
1: -188.6193237, 761.8539429, -211.5321503, 852.0767822, -1040.6960449, 973.3860474
2: -192.9447327, 754.8780518, -215.8921051, 843.9248047, -1036.8693848, 970.7701416
3: -304.6935730, 791.4774170, -341.7179871, 885.7481079, -1190.4416504, 1133.1954346
4: -307.6691284, 755.6786499, -344.5585022, 844.9055176, -1152.5747070, 1100.2370605

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013455, upper bound: 853.3014885
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024590, upper bound: 853.3024948
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024948, upper bound: 853.3024948
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -148.8383026, 661.7344971, -180.1655731, 801.5324097, -950.3706665, 841.9000854
1: -186.3859711, 753.2599487, -225.4790955, 912.3767090, -1098.7626953, 978.7390137
2: -190.7019958, 746.4499512, -230.5771484, 903.7211914, -1094.4230957, 977.0270996
3: -301.1500854, 782.5203247, -364.6372986, 947.7578735, -1248.9079590, 1147.1575928
4: -303.9338684, 747.4294434, -368.4224854, 904.2799072, -1208.2137451, 1115.8516846

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010657, upper bound: 853.3017303
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000126, upper bound: 853.3004878
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -156.2597046, 692.8746948, -179.5024261, 798.5998535, -954.8595581, 872.3770752
1: -195.5993347, 788.7254639, -224.6526794, 909.0396118, -1104.6389160, 1013.3781738
2: -200.0496063, 781.5624390, -229.7332153, 900.4190063, -1100.4686279, 1011.2955933
3: -315.8252258, 819.6289673, -363.2958374, 944.2952271, -1260.1204834, 1182.9246826
4: -318.9731750, 782.5482178, -367.0778198, 900.9774780, -1219.9504395, 1149.6259766

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010319, upper bound: 853.3014569
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999784, upper bound: 853.3002143
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -148.8383026, 661.7344971, -195.9449615, 868.3510132, -1017.1893311, 857.6793823
1: -186.3859711, 753.2599487, -244.9770966, 988.3905029, -1174.7761230, 998.2369995
2: -190.7019958, 746.4499512, -249.9883881, 978.9094238, -1169.6114502, 996.4383545
3: -301.1500854, 782.5203247, -396.1129761, 1027.1856689, -1328.3356934, 1178.6333008
4: -303.9338684, 747.4294434, -399.4000549, 979.5213013, -1283.4550781, 1146.8294678

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010657, upper bound: 853.3016036
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016642, upper bound: 853.3016712
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3021826, upper bound: 853.3023540
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -156.2597046, 692.8746948, -195.3118896, 865.5563965, -1021.8161011, 888.1865845
1: -195.5993347, 788.7254639, -244.1906433, 985.2094727, -1180.8088379, 1032.9161377
2: -200.0496063, 781.5624390, -249.1823425, 975.7633057, -1175.8128662, 1030.7446289
3: -315.8252258, 819.6289673, -394.8383789, 1023.8824463, -1339.7076416, 1214.4672852
4: -318.9731750, 782.5482178, -398.1156311, 976.3765869, -1295.3497314, 1180.6638184

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023795, upper bound: 853.3021591
time: 0.95 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.89 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 0, lower bound: -853.3023795, upper bound: 853.3021591
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -182.7681427, 808.7180176, -172.6161652, 763.3468018, -946.1149292, 981.3341064
1: -228.5348969, 920.6278076, -215.8617249, 868.9192505, -1097.4539795, 1136.4895020
2: -233.1255188, 911.7817383, -220.2556305, 860.7608643, -1093.8863525, 1132.0373535
3: -369.3415222, 956.9161377, -348.6884155, 903.1724243, -1272.5139160, 1305.6043701
4: -372.4694519, 912.2335815, -351.4360046, 861.5000000, -1233.9692383, 1263.6694336

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
time: 0.68 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
time: 0.71 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -181.3735657, 802.6942139, -203.9094696, 903.2384644, -1084.6119385, 1006.6036377
1: -226.7870789, 913.7738037, -254.9292297, 1028.1209717, -1254.9080811, 1168.7028809
2: -231.3600006, 904.9996948, -260.1297302, 1018.3677979, -1249.7277832, 1165.1293945
3: -366.5259094, 949.7840576, -412.1533813, 1068.4002686, -1434.9261475, 1361.9375000
4: -369.6641541, 905.4392090, -415.5236206, 1018.7225342, -1388.3867188, 1320.9628906

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006474, upper bound: 853.3000244
time: 0.83 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020275, upper bound: 853.3020275
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.62 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -853.3006474, upper bound: 853.3000244
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -853.3020275, upper bound: 853.3020275

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -172.6161652, 763.3468018, -172.6161652, 763.3468018, -935.9628296, 935.9628296
1: -215.8617249, 868.9192505, -215.8617249, 868.9192505, -1084.7807617, 1084.7807617
2: -220.2556305, 860.7608643, -220.2556305, 860.7608643, -1081.0164795, 1081.0164795
3: -348.6884155, 903.1724243, -348.6884155, 903.1724243, -1251.8608398, 1251.8608398
4: -351.4360046, 861.5000000, -351.4360046, 861.5000000, -1212.9357910, 1212.9357910

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023795, upper bound: 853.3021544
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023036, upper bound: 853.3021544
time: 0.72 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -203.9094696, 903.2384644, -172.6161652, 763.3468018, -967.2561646, 1075.8546143
1: -254.9292297, 1028.1209717, -215.8617249, 868.9192505, -1123.8483887, 1243.9826660
2: -260.1297302, 1018.3677979, -220.2556305, 860.7608643, -1120.8906250, 1238.6234131
3: -412.1533813, 1068.4002686, -348.6884155, 903.1724243, -1315.3258057, 1417.0886230
4: -415.5236206, 1018.7225342, -351.4360046, 861.5000000, -1277.0235596, 1370.1585693

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023795, upper bound: 853.3021544
time: 0.74 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023035, upper bound: 853.3021544
time: 0.78 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -180.1403351, 797.0331421, -197.8976135, 871.8310547, -1051.9710693, 994.9307861
1: -225.2473907, 907.3279419, -247.3440247, 992.4859009, -1217.7330322, 1154.6719971
2: -229.7859497, 898.6484375, -252.2371979, 983.4658203, -1213.2517090, 1150.8854980
3: -363.9935913, 943.1076660, -399.0464478, 1031.4986572, -1395.4921875, 1342.1540527
4: -367.1024170, 899.1166382, -402.1404419, 983.9295654, -1351.0319824, 1301.2570801

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005847, upper bound: 853.2999997
time: 0.76 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005737, upper bound: 853.2997641
time: 0.71 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -181.3735657, 802.6942139, -203.8071899, 902.7960815, -1084.1694336, 1006.5014038
1: -226.7870789, 913.7738037, -254.8004913, 1027.6179199, -1254.4050293, 1168.5742188
2: -231.3600006, 904.9996948, -259.9992065, 1017.8665771, -1249.2265625, 1164.9989014
3: -366.5259094, 949.7840576, -411.9476318, 1067.8769531, -1434.4028320, 1361.7315674
4: -369.6641541, 905.4392090, -415.3203125, 1018.2191772, -1387.8833008, 1320.7595215

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000244, upper bound: 853.3006474
time: 0.90 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000244, upper bound: 853.3020275
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.81 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3023795, upper bound: 853.3021544
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3023036, upper bound: 853.3021544
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3023795, upper bound: 853.3021544
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3023035, upper bound: 853.3021544
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3005847, upper bound: 853.2999997
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3005737, upper bound: 853.2997641
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3000244, upper bound: 853.3006474
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.81
Output dim: 0, lower bound: -853.3000244, upper bound: 853.3020275

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -169.5297699, 750.1862183, -901.9470825, 843.7593384
1: -190.0204926, 767.5302734, -212.0260468, 853.9455566, -1043.9660645, 979.5562134
2: -194.3727112, 760.5083618, -216.3945007, 845.8706055, -1040.2432861, 976.9027710
3: -306.9681091, 797.3639526, -342.5488281, 887.5798340, -1194.5479736, 1139.9128418
4: -309.9315796, 761.3359985, -345.3193970, 846.5714722, -1156.5029297, 1106.6553955

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -168.9175110, 747.3393555, -912.0208130, 897.9251099
1: -205.9361420, 829.8331299, -211.2400208, 850.6990967, -1056.6352539, 1041.0731201
2: -210.1320648, 821.9001465, -215.5415344, 842.6438599, -1052.7757568, 1037.4416504
3: -332.7737427, 862.5795288, -341.2768250, 884.2551270, -1217.0288086, 1203.8560791
4: -335.4803162, 822.6964111, -343.9974060, 843.4346924, -1178.9147949, 1166.6938477

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
time: 0.77 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -180.2634430, 801.9572144, -169.5297699, 750.1862183, -930.4496460, 971.4869385
1: -225.6020966, 912.8598022, -212.0260468, 853.9455566, -1079.5476074, 1124.8857422
2: -230.7018127, 904.2016602, -216.3945007, 845.8706055, -1076.5723877, 1120.5961914
3: -364.8340759, 948.2597046, -342.5488281, 887.5798340, -1252.4139404, 1290.8083496
4: -368.6170349, 904.7614136, -345.3193970, 846.5714722, -1215.1883545, 1250.0808105

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003752, upper bound: 853.3007444
time: 0.87 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023782, upper bound: 853.3021111
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -196.0472412, 868.7943115, -168.9175110, 747.3393555, -943.3865967, 1037.7117920
1: -245.1055756, 988.8947754, -211.2400208, 850.6990967, -1095.8044434, 1200.1347656
2: -250.1186981, 979.4112549, -215.5415344, 842.6438599, -1092.7624512, 1194.9525146
3: -396.3183899, 1027.7099609, -341.2768250, 884.2551270, -1280.5734863, 1368.9866943
4: -399.6038513, 980.0239868, -343.9974060, 843.4346924, -1243.0385742, 1324.0211182

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023035, upper bound: 853.3021544
time: 0.75 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023036, upper bound: 853.3021544
time: 1.10 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -177.0561371, 783.8473511, -175.5885162, 776.4868164, -953.5427856, 959.4357300
1: -221.4139709, 892.3302612, -219.7106323, 883.9606934, -1105.3743896, 1112.0408936
2: -225.9286346, 883.7355957, -224.4729156, 876.0089111, -1101.9375000, 1108.2084961
3: -357.8495483, 927.4904175, -354.5040283, 918.4062500, -1276.2556152, 1281.9943848
4: -360.9965820, 884.1587524, -357.8605652, 876.7702637, -1237.7668457, 1242.0192871

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
time: 0.85 seconds

## Relational analysis of IS_B2_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2999997
time: 0.80 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -176.4023285, 780.8722534, -190.4205170, 839.2810669, -1015.6833496, 971.2927856
1: -220.5764771, 888.9309082, -238.0125122, 955.4159546, -1175.9923096, 1126.9432373
2: -225.0192871, 880.3577881, -242.7419586, 946.6467285, -1171.6656494, 1123.0997314
3: -356.5047302, 924.0052490, -384.0323486, 993.0491943, -1349.5538330, 1308.0375977
4: -359.5909119, 880.8654175, -387.0486145, 947.3689575, -1306.9598389, 1267.9140625

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005737, upper bound: 853.2997641
time: 0.82 seconds

## Relational analysis of IS_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005737, upper bound: 853.2997641
time: 0.83 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -167.8638306, 738.4755859, -200.7057190, 888.0632935, -1055.9270020, 939.1812744
1: -209.8532562, 840.6962891, -250.9567871, 1010.8604736, -1220.7137451, 1091.6530762
2: -214.0453186, 833.1791992, -256.0322571, 1001.3695068, -1215.4147949, 1089.2114258
3: -338.2551880, 874.1419678, -405.5472717, 1050.6368408, -1388.8919678, 1279.6888428
4: -341.1071167, 834.0988159, -408.8330688, 1001.9187622, -1343.0255127, 1242.9318848

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
time: 0.85 seconds

## Relational analysis of IS_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983528, upper bound: 853.3005737
time: 0.87 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -181.2670288, 802.2296753, -203.8071899, 902.7960815, -1084.0628662, 1006.0368652
1: -226.6527863, 913.2465210, -254.8004913, 1027.6179199, -1254.2703857, 1168.0469971
2: -231.2234802, 904.4740601, -259.9992065, 1017.8665771, -1249.0900879, 1164.4731445
3: -366.3114319, 949.2357178, -411.9476318, 1067.8769531, -1434.1881104, 1361.1833496
4: -369.4518433, 904.9100952, -415.3203125, 1018.2191772, -1387.6710205, 1320.2304688

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.3020274
time: 0.80 seconds

## Relational analysis of IS_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983528, upper bound: 853.3019622
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.09 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3024373, upper bound: 853.3024373
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3003752, upper bound: 853.3007444
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3023782, upper bound: 853.3021111
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3023035, upper bound: 853.3021544
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3023036, upper bound: 853.3021544
IS_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
IS_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2999997
IS_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3005737, upper bound: 853.2997641
IS_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.3005737, upper bound: 853.2997641
IS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
IS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.2983528, upper bound: 853.3005737
IS_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.2983741, upper bound: 853.3020274
IS_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 0, lower bound: -853.2983528, upper bound: 853.3019622

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -151.7609406, 674.2295532, -825.9904785, 825.9904175
1: -190.0204926, 767.5302734, -190.0204926, 767.5302734, -957.5507812, 957.5507812
2: -194.3727112, 760.5083618, -194.3727112, 760.5083618, -954.8811035, 954.8811035
3: -306.9681091, 797.3639526, -306.9681091, 797.3639526, -1104.3320312, 1104.3320312
4: -309.9315796, 761.3359985, -309.9315796, 761.3359985, -1071.2674561, 1071.2674561

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024195, upper bound: 853.3024100
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024100
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -164.6814880, 729.0076294, -880.7685547, 838.9109497
1: -190.0204926, 767.5302734, -205.9361420, 829.8331299, -1019.8536377, 973.4664307
2: -194.3727112, 760.5083618, -210.1320648, 821.9001465, -1016.2728271, 970.6404419
3: -306.9681091, 797.3639526, -332.7737427, 862.5795288, -1169.5476074, 1130.1375732
4: -309.9315796, 761.3359985, -335.4803162, 822.6964111, -1132.6279297, 1096.8161621

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024195, upper bound: 853.3024100
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024100
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -151.7609406, 674.2295532, -838.9110107, 880.7684937
1: -205.9361420, 829.8331299, -190.0204926, 767.5302734, -973.4664307, 1019.8536377
2: -210.1320648, 821.9001465, -194.3727112, 760.5083618, -970.6404419, 1016.2728271
3: -332.7737427, 862.5795288, -306.9681091, 797.3639526, -1130.1376953, 1169.5474854
4: -335.4803162, 822.6964111, -309.9315796, 761.3359985, -1096.8161621, 1132.6279297

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024125
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -164.6814880, 729.0076294, -164.6814880, 729.0076294, -893.6890869, 893.6890869
1: -205.9361420, 829.8331299, -205.9361420, 829.8331299, -1035.7691650, 1035.7691650
2: -210.1320648, 821.9001465, -210.1320648, 821.9001465, -1032.0322266, 1032.0322266
3: -332.7737427, 862.5795288, -332.7737427, 862.5795288, -1195.3529053, 1195.3529053
4: -335.4803162, 822.6964111, -335.4803162, 822.6964111, -1158.1767578, 1158.1767578

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024125
time: 1.25 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -175.5865479, 776.4785767, -168.3283691, 744.6633301, -920.2497559, 944.8069458
1: -219.7080994, 883.9515991, -210.5255280, 847.6616821, -1067.3695068, 1094.4768066
2: -224.4704285, 875.9996948, -214.8618774, 839.6724854, -1064.1429443, 1090.8614502
3: -354.5000610, 918.3965454, -340.0801697, 881.0695801, -1235.5694580, 1258.4766846
4: -357.8567200, 876.7610474, -342.8297424, 840.3947754, -1198.2514648, 1219.5908203

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003752, upper bound: 853.3007444
time: 0.80 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003752, upper bound: 853.3007444
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -180.1510773, 801.4645996, -169.5297699, 750.1862183, -930.3372803, 970.9943848
1: -225.4611816, 912.2997437, -212.0260468, 853.9455566, -1079.4067383, 1124.3258057
2: -230.5589142, 903.6455688, -216.3945007, 845.8706055, -1076.4293213, 1120.0400391
3: -364.6074524, 947.6780396, -342.5488281, 887.5798340, -1252.1872559, 1290.2268066
4: -368.3922119, 904.2051392, -345.3193970, 846.5714722, -1214.9633789, 1249.5245361

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023782, upper bound: 853.3021111
time: 0.90 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023782, upper bound: 853.3021111
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -196.0472412, 868.7943115, -151.7609406, 674.2295532, -870.2767334, 1020.5551758
1: -245.1055756, 988.8947754, -190.0204926, 767.5302734, -1012.6358032, 1178.9152832
2: -250.1186981, 979.4112549, -194.3727112, 760.5083618, -1010.6270752, 1173.7835693
3: -396.3183899, 1027.7099609, -306.9681091, 797.3639526, -1193.6823730, 1334.6781006
4: -399.6038513, 980.0239868, -309.9315796, 761.3359985, -1160.9396973, 1289.9554443

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000974, upper bound: 853.3007322
time: 0.85 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023014, upper bound: 853.3021111
time: 1.14 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -196.0472412, 868.7943115, -164.6814880, 729.0076294, -925.0548706, 1033.4758301
1: -245.1055756, 988.8947754, -205.9361420, 829.8331299, -1074.9384766, 1194.8309326
2: -250.1186981, 979.4112549, -210.1320648, 821.9001465, -1072.0187988, 1189.5430908
3: -396.3183899, 1027.7099609, -332.7737427, 862.5795288, -1258.8979492, 1360.4836426
4: -399.6038513, 980.0239868, -335.4803162, 822.6964111, -1222.3002930, 1315.5041504

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000974, upper bound: 853.3007322
time: 0.89 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023014, upper bound: 853.3021111
time: 1.03 seconds

## BFS IS instance: IS_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -164.7101746, 724.8456421, -175.1410065, 774.3479004, -939.0581055, 899.9864502
1: -205.9346466, 825.1923218, -219.1485748, 881.5332031, -1087.4675293, 1044.3406982
2: -210.0892487, 817.7857056, -223.9011536, 873.6163940, -1083.7055664, 1041.6868896
3: -331.9472656, 858.0164185, -353.5676575, 915.8997192, -1247.8469238, 1211.5841064
4: -334.8049316, 818.7171631, -356.9242859, 874.3875732, -1209.1925049, 1175.6414795

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1_A1_A1

### Relational analysis result of IS_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
time: 0.78 seconds

## Relational analysis of IS_B2_B1_B1_A1_A2

### Relational analysis result of IS_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
time: 0.81 seconds

## BFS IS instance: IS_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -177.6561127, 786.5600586, -175.5885162, 776.4868164, -954.1429443, 962.1484985
1: -222.1611938, 895.4195557, -219.7106323, 883.9606934, -1106.1217041, 1115.1301270
2: -226.6896210, 886.7830811, -224.4729156, 876.0089111, -1102.6984863, 1111.2559814
3: -359.0762634, 930.6970825, -354.5040283, 918.4062500, -1277.4824219, 1285.2011719
4: -362.2252502, 887.1995850, -357.8605652, 876.7702637, -1238.9954834, 1245.0601807

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1_A2_A1

### Relational analysis result of IS_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2999997
time: 1.44 seconds

## Relational analysis of IS_B2_B1_B1_A2_A2

### Relational analysis result of IS_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2999997
time: 0.80 seconds

## BFS IS instance: IS_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -158.8666687, 706.3535767, -190.4205170, 839.2810669, -998.1477051, 896.7741089
1: -198.9067993, 804.1928711, -238.0125122, 955.4159546, -1154.3226318, 1042.2050781
2: -203.3928070, 796.6164551, -242.7419586, 946.6467285, -1150.0395508, 1039.3583984
3: -321.5345154, 835.4827271, -384.0323486, 993.0491943, -1314.5837402, 1219.5150146
4: -324.8881226, 797.1658936, -387.0486145, 947.3689575, -1272.2570801, 1184.2144775

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_B2_A1_A1

### Relational analysis result of IS_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2983528
time: 0.75 seconds

## Relational analysis of IS_B2_B1_B2_A1_A2

### Relational analysis result of IS_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2997641
time: 0.82 seconds

## BFS IS instance: IS_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -172.0792999, 762.1834717, -190.4205170, 839.2810669, -1011.3603516, 952.6040039
1: -215.1632080, 867.6591187, -238.0125122, 955.4159546, -1170.5788574, 1105.6713867
2: -219.4976349, 859.2055664, -242.7419586, 946.6467285, -1166.1442871, 1101.9475098
3: -347.8282471, 901.9058838, -384.0323486, 993.0491943, -1340.8774414, 1285.9382324
4: -350.9017639, 859.7211914, -387.0486145, 947.3689575, -1298.2706299, 1246.7697754

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2983528
time: 0.88 seconds

## Relational analysis of IS_B2_B1_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2997641
time: 0.76 seconds

## BFS IS instance: IS_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -164.7101746, 724.8456421, -177.2362823, 787.8237915, -952.5339355, 902.0818481
1: -205.9346466, 825.1923218, -221.8335114, 896.8062134, -1102.7407227, 1047.0256348
2: -210.0892487, 817.7857056, -226.8646088, 888.3746948, -1098.4638672, 1044.6502686
3: -331.9472656, 858.0164185, -358.5958252, 931.6967773, -1263.6439209, 1216.6121826
4: -334.8049316, 818.7171631, -362.3752441, 889.0526733, -1223.8576660, 1181.0924072

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A1_B1_A1

### Relational analysis result of IS_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
time: 0.71 seconds

## Relational analysis of IS_B2_B2_A1_B1_A2

### Relational analysis result of IS_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
time: 0.77 seconds

## BFS IS instance: IS_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -163.7888947, 720.9832153, -192.6842194, 852.9357910, -1016.7246094, 913.6674194
1: -204.7572021, 820.8000488, -240.9305115, 970.8686523, -1175.6258545, 1061.7305908
2: -208.8747559, 813.3543091, -245.8115692, 961.6321411, -1170.5067139, 1059.1657715
3: -330.0867920, 853.4485474, -389.3929749, 1009.1465454, -1339.2333984, 1242.8415527
4: -332.9851074, 814.2690430, -392.6026611, 962.4205322, -1295.4053955, 1206.8717041

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A1_B2_A1

### Relational analysis result of IS_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
time: 0.80 seconds

## Relational analysis of IS_B2_B2_A1_B2_A2

### Relational analysis result of IS_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
time: 1.15 seconds

## BFS IS instance: IS_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -178.2237549, 789.2321167, -180.1647186, 801.5281372, -979.7518311, 969.3967896
1: -222.8706512, 898.4615479, -225.4780121, 912.3720703, -1135.2426758, 1123.9393311
2: -227.4176331, 889.7716064, -230.5760498, 903.7166748, -1131.1342773, 1120.3474121
3: -360.2528381, 933.8392944, -364.6354675, 947.7528687, -1308.0057373, 1298.4744873
4: -363.4254456, 890.1657715, -368.4206848, 904.2753296, -1267.7008057, 1258.5864258

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A2_B1_A1

### Relational analysis result of IS_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019352, upper bound: 853.3020274
time: 0.80 seconds

## Relational analysis of IS_B2_B2_A2_B1_A2

### Relational analysis result of IS_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019352, upper bound: 853.3020062
time: 0.81 seconds

## BFS IS instance: IS_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -177.5394135, 786.1086426, -195.9453735, 868.3528442, -1045.8920898, 982.0538940
1: -221.9956512, 894.8942871, -244.9775391, 988.3927612, -1210.3884277, 1139.8717041
2: -226.4705200, 886.2291870, -249.9889069, 978.9113159, -1205.3818359, 1136.2180176
3: -358.8446655, 930.1800537, -396.1136169, 1027.1881104, -1386.0327148, 1326.2937012
4: -361.9570618, 886.7067261, -399.4012146, 979.5230713, -1341.4801025, 1286.1079102

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_A2_B2_A1

### Relational analysis result of IS_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019605, upper bound: 853.3019622
time: 0.75 seconds

## Relational analysis of IS_B2_B2_A2_B2_A2

### Relational analysis result of IS_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019605, upper bound: 853.3019622
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.72 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024195, upper bound: 853.3024100
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024100
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024195, upper bound: 853.3024100
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024100
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024125
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024125
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
IS_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3003752, upper bound: 853.3007444
IS_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3003752, upper bound: 853.3007444
IS_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3023782, upper bound: 853.3021111
IS_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3023782, upper bound: 853.3021111
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3000974, upper bound: 853.3007322
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3023014, upper bound: 853.3021111
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3000974, upper bound: 853.3007322
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3023014, upper bound: 853.3021111
IS_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
IS_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2985889
IS_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2999997
IS_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983741, upper bound: 853.2999997
IS_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2983528
IS_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2997641
IS_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2983528
IS_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2983528, upper bound: 853.2997641
IS_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3019352, upper bound: 853.3020274
IS_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3019352, upper bound: 853.3020062
IS_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3019605, upper bound: 853.3019622
IS_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 0, lower bound: -853.3019605, upper bound: 853.3019622

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -148.8383026, 661.7344971, -813.4953613, 823.0678101
1: -190.0204926, 767.5302734, -186.3859711, 753.2599487, -943.2804565, 953.9161987
2: -194.3727112, 760.5083618, -190.7019958, 746.4499512, -940.8226318, 951.2103271
3: -306.9681091, 797.3639526, -301.1500854, 782.5203247, -1089.4884033, 1098.5140381
4: -309.9315796, 761.3359985, -303.9338684, 747.4294434, -1057.3608398, 1065.2698975

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024152, upper bound: 853.3024152
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024152, upper bound: 853.3024152
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -149.9294434, 666.0076904, -156.2597046, 692.8746948, -842.8040771, 822.2673950
1: -187.7208252, 758.1879272, -195.5993347, 788.7254639, -976.4462891, 953.7872314
2: -192.0246735, 751.2490845, -200.0496063, 781.5624390, -973.5870972, 951.2987061
3: -303.2331848, 787.6768799, -315.8252258, 819.6289673, -1122.8621826, 1103.5019531
4: -306.2028809, 752.0386353, -318.9731750, 782.5482178, -1088.7510986, 1071.0117188

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020593, upper bound: 853.3022925
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024153
time: 0.79 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -151.7609406, 674.2295532, -161.5120239, 715.4299927, -867.1908569, 835.7414551
1: -190.0204926, 767.5302734, -202.0000610, 814.3376465, -1004.3581543, 969.5302734
2: -194.3727112, 760.5083618, -206.1473389, 806.6093750, -1000.9820557, 966.6557007
3: -306.9681091, 797.3639526, -326.4761658, 846.4689941, -1153.4371338, 1123.8399658
4: -309.9315796, 761.3359985, -329.0003357, 807.5625000, -1117.4940186, 1090.3361816

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020708, upper bound: 853.3022871
time: 0.77 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024195, upper bound: 853.3024100
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -149.9294434, 666.0076904, -169.2769623, 748.9935913, -898.9230347, 835.2846069
1: -187.7208252, 758.1879272, -211.6526184, 852.5535278, -1040.2742920, 969.8405762
2: -192.0246735, 751.2490845, -216.0144806, 844.3994141, -1036.4240723, 967.2635498
3: -303.2331848, 787.6768799, -341.9109802, 886.2436523, -1189.4768066, 1129.5878906
4: -306.2028809, 752.0386353, -344.7489624, 845.3828735, -1151.5855713, 1096.7875977

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020593, upper bound: 853.3022925
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024100
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -161.5120239, 715.4299927, -151.7609406, 674.2295532, -835.7414551, 867.1908569
1: -202.0000610, 814.3376465, -190.0204926, 767.5302734, -969.5303345, 1004.3581543
2: -206.1473389, 806.6093750, -194.3727112, 760.5083618, -966.6557007, 1000.9820557
3: -326.4761658, 846.4689941, -306.9681091, 797.3639526, -1123.8399658, 1153.4371338
4: -329.0003357, 807.5625000, -309.9315796, 761.3359985, -1090.3361816, 1117.4940186

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022871, upper bound: 853.3020708
time: 0.75 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024195
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -169.2769623, 748.9935913, -149.9294434, 666.0076904, -835.2846069, 898.9230347
1: -211.6526184, 852.5535278, -187.7208252, 758.1879272, -969.8405762, 1040.2742920
2: -216.0144806, 844.3994141, -192.0246735, 751.2490845, -967.2635498, 1036.4240723
3: -341.9109802, 886.2436523, -303.2331848, 787.6768799, -1129.5878906, 1189.4768066
4: -344.7489624, 845.3828735, -306.2028809, 752.0386353, -1096.7875977, 1151.5856934

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022925, upper bound: 853.3020593
time: 0.71 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024153
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -161.5120239, 715.4299927, -164.6814880, 729.0076294, -890.5195312, 880.1114502
1: -202.0000610, 814.3376465, -205.9361420, 829.8331299, -1031.8331299, 1020.2738037
2: -206.1473389, 806.6093750, -210.1320648, 821.9001465, -1028.0474854, 1016.7414551
3: -326.4761658, 846.4689941, -332.7737427, 862.5795288, -1189.0552979, 1179.2426758
4: -329.0003357, 807.5625000, -335.4803162, 822.6964111, -1151.6967773, 1143.0427246

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
time: 0.79 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -169.2769623, 748.9935913, -163.3690033, 723.0755005, -892.3524170, 912.3625488
1: -211.6526184, 852.5535278, -204.2965240, 823.0869141, -1034.7395020, 1056.8498535
2: -216.0144806, 844.3994141, -208.4610443, 815.2279053, -1031.2424316, 1052.8603516
3: -341.9109802, 886.2436523, -330.0932312, 855.5851440, -1197.4960938, 1216.3366699
4: -344.7489624, 845.3828735, -332.7918091, 816.0349121, -1160.7839355, 1178.1746826

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022925, upper bound: 853.3020319
time: 0.76 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -175.5865479, 776.4785767, -150.4915619, 668.4534302, -844.0397949, 926.9700317
1: -219.7080994, 883.9515991, -188.4345093, 760.9624023, -980.6705322, 1072.3861084
2: -224.4704285, 875.9996948, -192.7486572, 754.0193481, -978.4897461, 1068.7481689
3: -354.5000610, 918.3965454, -304.3785706, 790.5622559, -1145.0620117, 1222.7751465
4: -357.8567200, 876.7610474, -307.3188782, 754.8644409, -1112.7209473, 1184.0799561

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003629, upper bound: 853.3007444
time: 1.01 seconds

## Relational analysis of IS_B1_A2_A1_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003383, upper bound: 853.3007440
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -175.5865479, 776.4785767, -163.4738617, 723.4833984, -899.0698242, 939.9524536
1: -219.7080994, 883.9515991, -204.4270477, 823.5458374, -1043.2539062, 1088.3782959
2: -224.4704285, 875.9996948, -208.5911102, 815.6982422, -1040.1687012, 1084.5906982
3: -354.5000610, 918.3965454, -330.2944031, 856.0652466, -1210.5651855, 1248.6909180
4: -357.8567200, 876.7610474, -332.9816284, 816.5159912, -1174.3725586, 1209.7426758

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003629, upper bound: 853.3007444
time: 0.90 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003383, upper bound: 853.3007440
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -180.1435852, 801.4296265, -151.7609406, 674.2295532, -854.3729858, 953.1905518
1: -225.4519348, 912.2600098, -190.0204926, 767.5302734, -992.9821777, 1102.2805176
2: -230.5494843, 903.6065674, -194.3727112, 760.5083618, -991.0578613, 1097.9792480
3: -364.5921021, 947.6369019, -306.9681091, 797.3639526, -1161.9560547, 1254.6049805
4: -368.3765259, 904.1666260, -309.9315796, 761.3359985, -1129.7122803, 1214.0981445

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A2_B1_B1

### Relational analysis result of IS_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023631, upper bound: 853.3021111
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A1_A2_B1_B2

### Relational analysis result of IS_B1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023371, upper bound: 853.3021111
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -180.1489258, 801.4544678, -164.6814880, 729.0076294, -909.1565552, 966.1359863
1: -225.4585114, 912.2882690, -205.9361420, 829.8331299, -1055.2916260, 1118.2243652
2: -230.5561981, 903.6341553, -210.1320648, 821.9001465, -1052.4562988, 1113.7662354
3: -364.6029358, 947.6662598, -332.7737427, 862.5795288, -1227.1823730, 1280.4399414
4: -368.3877563, 904.1939087, -335.4803162, 822.6964111, -1191.0842285, 1239.6741943

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_A2_B2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023631, upper bound: 853.3021111
time: 0.82 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023370, upper bound: 853.3021111
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -150.4992676, 668.4890747, -858.9096069, 989.7803345
1: -238.0125122, 955.4159546, -188.4441376, 761.0029907, -999.0155029, 1143.8601074
2: -242.7419586, 946.6467285, -192.7585144, 754.0593262, -996.8012695, 1139.4050293
3: -384.0323486, 993.0491943, -304.3944092, 790.6040649, -1174.6364746, 1297.4436035
4: -387.0486145, 947.3689575, -307.3350525, 754.9039917, -1141.9526367, 1254.7038574

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000849, upper bound: 853.3007322
time: 0.86 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000606, upper bound: 853.3007322
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -195.9145355, 868.2098389, -151.7609406, 674.2295532, -870.1441040, 1019.9707031
1: -244.9395294, 988.2302856, -190.0204926, 767.5302734, -1012.4697876, 1178.2507324
2: -249.9499817, 978.7516479, -194.3727112, 760.5083618, -1010.4583130, 1173.1241455
3: -396.0508423, 1027.0200195, -306.9681091, 797.3639526, -1193.4147949, 1333.9881592
4: -399.3372498, 979.3645630, -309.9315796, 761.3359985, -1160.6730957, 1289.2960205

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B1_A2_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015349, upper bound: 853.3016261
time: 0.81 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022641, upper bound: 853.3021161
time: 0.84 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -190.4205170, 839.2810669, -163.4815979, 723.5194092, -913.9399414, 1002.7626953
1: -238.0125122, 955.4159546, -204.4366608, 823.5867920, -1061.5993652, 1159.8522949
2: -242.7419586, 946.6467285, -208.6009369, 815.7383423, -1058.4803467, 1155.2474365
3: -384.0323486, 993.0491943, -330.3103333, 856.1076050, -1240.1398926, 1323.3594971
4: -387.0486145, 947.3689575, -332.9978333, 816.5560913, -1203.6044922, 1280.3668213

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000849, upper bound: 853.3007322
time: 0.77 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000606, upper bound: 853.3007322
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -195.9213562, 868.2412720, -164.6814880, 729.0076294, -924.9288940, 1032.9226074
1: -244.9479065, 988.2661133, -205.9361420, 829.8331299, -1074.7810059, 1194.2021484
2: -249.9585724, 978.7866211, -210.1320648, 821.9001465, -1071.8585205, 1188.9183350
3: -396.0646362, 1027.0571289, -332.7737427, 862.5795288, -1258.6440430, 1359.8306885
4: -399.3513794, 979.3992310, -335.4803162, 822.6964111, -1222.0478516, 1314.8795166

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022756, upper bound: 853.3021111
time: 1.22 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022614, upper bound: 853.3021111
time: 0.86 seconds

## BFS IS instance: IS_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -147.2155609, 650.5834351, -175.1032867, 774.1677246, -921.3832397, 825.6867065
1: -184.2910004, 740.7362671, -219.1014557, 881.3285522, -1065.6192627, 959.8377075
2: -188.3867950, 734.1943970, -223.8530579, 873.4149780, -1061.8017578, 958.0474854
3: -297.1069031, 769.8516235, -353.4891663, 915.6884766, -1212.7954102, 1123.3408203
4: -300.1649475, 735.2525635, -356.8452148, 874.1873169, -1174.3522949, 1092.0977783

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976512, upper bound: 853.2979378
time: 0.78 seconds

## Relational analysis of IS_B2_B1_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977320, upper bound: 853.2978436
time: 0.83 seconds

## BFS IS instance: IS_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -159.8361969, 703.9571533, -175.1066132, 774.1807251, -934.0168457, 879.0636597
1: -199.8112335, 801.4399414, -219.1054688, 881.3434448, -1081.1546631, 1020.5454102
2: -203.8378906, 794.0743408, -223.8572540, 873.4298706, -1077.2678223, 1017.9315186
3: -322.1496582, 833.3248291, -353.4953613, 915.7041016, -1237.8537598, 1186.8201904
4: -325.0640564, 795.0015869, -356.8516235, 874.2020874, -1199.2661133, 1151.8532715

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976512, upper bound: 853.2979378
time: 0.73 seconds

## Relational analysis of IS_B2_B1_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977320, upper bound: 853.2978436
time: 0.76 seconds

## BFS IS instance: IS_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -159.3210297, 708.4607544, -175.5885162, 776.4868164, -935.8077393, 884.0491943
1: -199.4738159, 806.5970459, -219.7106323, 883.9606934, -1083.4343262, 1026.3076172
2: -203.9767761, 798.9792480, -224.4729156, 876.0089111, -1079.9857178, 1023.4519043
3: -322.4667664, 837.9702148, -354.5040283, 918.4062500, -1240.8729248, 1192.4742432
4: -325.8344421, 799.5161133, -357.8605652, 876.7702637, -1202.6047363, 1157.3767090

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998677, upper bound: 853.2994631
time: 0.79 seconds

## Relational analysis of IS_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005847, upper bound: 853.2999997
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -172.6493073, 764.7589111, -175.5885162, 776.4868164, -949.1361084, 940.3472900
1: -215.8741913, 870.5926514, -219.7106323, 883.9606934, -1099.8348389, 1090.3032227
2: -220.2227325, 862.1002197, -224.4729156, 876.0089111, -1096.2314453, 1086.5731201
3: -348.9945984, 904.9511719, -354.5040283, 918.4062500, -1267.4007568, 1259.4552002
4: -352.0704041, 862.6091919, -357.8605652, 876.7702637, -1228.8406982, 1220.4693604

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998677, upper bound: 853.2994631
time: 0.75 seconds

## Relational analysis of IS_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005847, upper bound: 853.2999997
time: 1.43 seconds

## BFS IS instance: IS_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -147.2155609, 650.5834351, -189.9106445, 836.9133911, -984.1289673, 840.4940796
1: -184.2910004, 740.7362671, -237.3800354, 952.7298584, -1137.0205078, 978.1162109
2: -188.3867950, 734.1943970, -242.0986786, 943.9910889, -1132.3775635, 976.2929688
3: -297.1069031, 769.8516235, -382.9841309, 990.2719727, -1287.3789062, 1152.8356934
4: -300.1649475, 735.2525635, -385.9889526, 944.7478027, -1244.9127197, 1121.2414551

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B2_A1_A1_A1

### Relational analysis result of IS_B2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985885, upper bound: 853.2983012
time: 0.84 seconds

## Relational analysis of IS_B2_B1_B2_A1_A1_A2

### Relational analysis result of IS_B2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985889, upper bound: 853.2983741
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -159.3290710, 708.4985352, -190.4205170, 839.2810669, -998.6100464, 898.9190674
1: -199.4838409, 806.6398926, -238.0125122, 955.4159546, -1154.8996582, 1044.6523438
2: -203.9870148, 799.0214233, -242.7419586, 946.6467285, -1150.6336670, 1041.7634277
3: -322.4833374, 838.0144653, -384.0323486, 993.0491943, -1315.5324707, 1222.0468750
4: -325.8513184, 799.5580444, -387.0486145, 947.3689575, -1273.2202148, 1186.6065674

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_B1_B2_A1_A2_A1

### Relational analysis result of IS_B2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985885, upper bound: 853.2997251
time: 0.87 seconds

## Relational analysis of IS_B2_B1_B2_A1_A2_A2

### Relational analysis result of IS_B2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2985889, upper bound: 853.2997641
time: 0.83 seconds

## BFS IS instance: IS_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -159.8361969, 703.9571533, -189.9144745, 836.9295044, -996.7656250, 893.8714600
1: -199.8112335, 801.4399414, -237.3846893, 952.7484741, -1152.5595703, 1038.8245850
2: -203.8378906, 794.0743408, -242.1035767, 944.0092163, -1147.8471680, 1036.1778564
3: -322.1496582, 833.3248291, -382.9914551, 990.2911987, -1312.4406738, 1216.3162842
4: -325.0640564, 795.0015869, -385.9965820, 944.7661133, -1269.8302002, 1180.9981689

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_B2_A2_A1_B1

### Relational analysis result of IS_B2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2970340, upper bound: 853.2975231
time: 0.87 seconds

## Relational analysis of IS_B2_B1_B2_A2_A1_B2

### Relational analysis result of IS_B2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2977320, upper bound: 853.2977320
time: 0.84 seconds

## BFS IS instance: IS_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -172.6582794, 764.8010254, -190.4205170, 839.2810669, -1011.9393311, 955.2214966
1: -215.8853912, 870.6405029, -238.0125122, 955.4159546, -1171.3010254, 1108.6527100
2: -220.2342224, 862.1472778, -242.7419586, 946.6467285, -1166.8807373, 1104.8892822
3: -349.0131836, 905.0006714, -384.0323486, 993.0491943, -1342.0623779, 1289.0329590
4: -352.0893860, 862.6558228, -387.0486145, 947.3689575, -1299.4583740, 1249.7044678

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982089, upper bound: 853.2988024
time: 0.74 seconds

## Relational analysis of IS_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2982171, upper bound: 853.2988064
time: 0.94 seconds

## BFS IS instance: IS_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -147.2155609, 650.5834351, -177.0948792, 787.1797485, -934.3953247, 827.6782227
1: -184.2910004, 740.7362671, -221.6573944, 896.0750122, -1080.3658447, 962.3935547
2: -188.3867950, 734.1943970, -226.6846161, 887.6512451, -1076.0380859, 960.8790283
3: -297.1069031, 769.8516235, -358.3077087, 930.9397583, -1228.0466309, 1128.1591797
4: -300.1649475, 735.2525635, -362.0858765, 888.3336182, -1188.4985352, 1097.3383789

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2976512, upper bound: 853.3000604
time: 0.81 seconds

## Relational analysis of IS_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991486, upper bound: 853.3000604
time: 1.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.76 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024152, upper bound: 853.3024152
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024152, upper bound: 853.3024152
IS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3020593, upper bound: 853.3022925
IS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024153
IS_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3020708, upper bound: 853.3022871
IS_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024195, upper bound: 853.3024100
IS_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3020593, upper bound: 853.3022925
IS_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024153, upper bound: 853.3024100
IS_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3022871, upper bound: 853.3020708
IS_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024195
IS_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3022925, upper bound: 853.3020593
IS_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024153
IS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
IS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
IS_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3022925, upper bound: 853.3020319
IS_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3024100, upper bound: 853.3024100
IS_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3003629, upper bound: 853.3007444
IS_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3003383, upper bound: 853.3007440
IS_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3003629, upper bound: 853.3007444
IS_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3003383, upper bound: 853.3007440
IS_B1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3023631, upper bound: 853.3021111
IS_B1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3023371, upper bound: 853.3021111
IS_B1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3023631, upper bound: 853.3021111
IS_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3023370, upper bound: 853.3021111
IS_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3000849, upper bound: 853.3007322
IS_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3000606, upper bound: 853.3007322
IS_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3015349, upper bound: 853.3016261
IS_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3022641, upper bound: 853.3021161
IS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3000849, upper bound: 853.3007322
IS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3000606, upper bound: 853.3007322
IS_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3022756, upper bound: 853.3021111
IS_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3022614, upper bound: 853.3021111
IS_B2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2976512, upper bound: 853.2979378
IS_B2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2977320, upper bound: 853.2978436
IS_B2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2976512, upper bound: 853.2979378
IS_B2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2977320, upper bound: 853.2978436
IS_B2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2998677, upper bound: 853.2994631
IS_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3005847, upper bound: 853.2999997
IS_B2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2998677, upper bound: 853.2994631
IS_B2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.3005847, upper bound: 853.2999997
IS_B2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2985885, upper bound: 853.2983012
IS_B2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2985889, upper bound: 853.2983741
IS_B2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2985885, upper bound: 853.2997251
IS_B2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2985889, upper bound: 853.2997641
IS_B2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2970340, upper bound: 853.2975231
IS_B2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2977320, upper bound: 853.2977320
IS_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2982089, upper bound: 853.2988024
IS_B2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2982171, upper bound: 853.2988064
IS_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2976512, upper bound: 853.3000604
IS_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.76
Output dim: 0, lower bound: -853.2991486, upper bound: 853.3000604
IS_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.2997641, upper bound: 853.3005737
IS_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.3019352, upper bound: 853.3020274
IS_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.3019352, upper bound: 853.3020062
IS_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.3019605, upper bound: 853.3019622
IS_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.76
Output dim: 0, lower bound: -853.3019605, upper bound: 853.3019622
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=991.4861450195312
rel_dist={0: [-853.3029989672884, 853.3029989672882]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1107.27 seconds
