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
execution time: IAR + LP analysis = 1.89 + 2.03 = 3.91 seconds
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
Binary search time: 90.55 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1105.53 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -853.3023633, upper bound: 853.3023633

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020242, upper bound: 853.3022118
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022118, upper bound: 853.3020242
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975997, upper bound: 853.2975997
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975997, upper bound: 853.2975997
time: 0.86 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.48 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -853.3020242, upper bound: 853.3022118
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -853.3022118, upper bound: 853.3020242
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -853.2975997, upper bound: 853.2975997
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 0, lower bound: -853.2975997, upper bound: 853.2975997

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014727, upper bound: 853.3015767
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014727, upper bound: 853.3015767
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020101, upper bound: 853.3020203
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022056, upper bound: 853.3019848
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2319316, upper bound: 853.2318482
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2319316, upper bound: 853.2318482
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975743, upper bound: 853.2975997
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975997, upper bound: 853.2975839
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.3014727, upper bound: 853.3015767
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.3014727, upper bound: 853.3015767
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.3020101, upper bound: 853.3020203
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.3022056, upper bound: 853.3019848
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.2319316, upper bound: 853.2318482
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.2319316, upper bound: 853.2318482
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.2975743, upper bound: 853.2975997
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -853.2975997, upper bound: 853.2975839

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014679, upper bound: 853.3015100
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014328, upper bound: 853.3014888
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014677, upper bound: 853.3015720
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014700, upper bound: 853.3014637
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019419, upper bound: 853.3018699
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019366, upper bound: 853.3019632
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014495, upper bound: 853.3012945
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013136, upper bound: 853.3012962
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2975743, upper bound: 853.2971791
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2970811, upper bound: 853.2975997
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960245, upper bound: 853.2960496
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2961202, upper bound: 853.2960496
time: 0.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3014679, upper bound: 853.3015100
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3014328, upper bound: 853.3014888
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3014677, upper bound: 853.3015720
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3014700, upper bound: 853.3014637
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3019419, upper bound: 853.3018699
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3019366, upper bound: 853.3019632
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3014495, upper bound: 853.3012945
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.3013136, upper bound: 853.3012962
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.2975743, upper bound: 853.2971791
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.2970811, upper bound: 853.2975997
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.2960245, upper bound: 853.2960496
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 0, lower bound: -853.2961202, upper bound: 853.2960496

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014674, upper bound: 853.3015047
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014679, upper bound: 853.3014328
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014308, upper bound: 853.3014856
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014308, upper bound: 853.3014450
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014528, upper bound: 853.3015720
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014677, upper bound: 853.3014338
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014202, upper bound: 853.3014509
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014572, upper bound: 853.3014258
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019184, upper bound: 853.3018699
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019386, upper bound: 853.3018699
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016749, upper bound: 853.3017585
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017314, upper bound: 853.3017388
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013073, upper bound: 853.3011216
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012252, upper bound: 853.3011216
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012525, upper bound: 853.3012525
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012857, upper bound: 853.3012525
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2317087, upper bound: 853.2315070
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2317087, upper bound: 853.2315153
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2314959, upper bound: 853.2315071
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2314959, upper bound: 853.2315674
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2959919, upper bound: 853.2959660
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2959660, upper bound: 853.2959731
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960245, upper bound: 853.2960245
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2960245, upper bound: 853.2960496
time: 0.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014674, upper bound: 853.3015047
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014679, upper bound: 853.3014328
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014308, upper bound: 853.3014856
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014308, upper bound: 853.3014450
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014528, upper bound: 853.3015720
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014677, upper bound: 853.3014338
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014202, upper bound: 853.3014509
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3014572, upper bound: 853.3014258
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3019184, upper bound: 853.3018699
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3019386, upper bound: 853.3018699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3016749, upper bound: 853.3017585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3017314, upper bound: 853.3017388
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3013073, upper bound: 853.3011216
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3012252, upper bound: 853.3011216
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3012525, upper bound: 853.3012525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.3012857, upper bound: 853.3012525
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2317087, upper bound: 853.2315070
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2317087, upper bound: 853.2315153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2314959, upper bound: 853.2315071
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2314959, upper bound: 853.2315674
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2959919, upper bound: 853.2959660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2959660, upper bound: 853.2959731
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2960245, upper bound: 853.2960245
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.41
Output dim: 0, lower bound: -853.2960245, upper bound: 853.2960496

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014625, upper bound: 853.3014694
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014213, upper bound: 853.3014978
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014274, upper bound: 853.3014085
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014085, upper bound: 853.3014085
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014181, upper bound: 853.3014732
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014181, upper bound: 853.3014181
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 1.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014330, upper bound: 853.3015014
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014528, upper bound: 853.3014661
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014204, upper bound: 853.3014259
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014204, upper bound: 853.3014209
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005732
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005546
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004503, upper bound: 853.3004461
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004503, upper bound: 853.3004461
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019116, upper bound: 853.3018669
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018669, upper bound: 853.3018669
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019386, upper bound: 853.3018699
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018699, upper bound: 853.3018699
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011919, upper bound: 853.3012255
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011919, upper bound: 853.3012255
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012421, upper bound: 853.3012009
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012421, upper bound: 853.3012009
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011216, upper bound: 853.3011216
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013073, upper bound: 853.3011216
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012015, upper bound: 853.3010769
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010874, upper bound: 853.3010769
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012437, upper bound: 853.3012303
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012303, upper bound: 853.3012303
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2959777, upper bound: 853.2959660
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2959919, upper bound: 853.2959660
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2916992, upper bound: 853.2916993
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2916992, upper bound: 853.2916993
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2915164, upper bound: 853.2914871
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2915286, upper bound: 853.2914871
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2958365, upper bound: 853.2958365
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2958365, upper bound: 853.2958365
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014625, upper bound: 853.3014694
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014213, upper bound: 853.3014978
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014274, upper bound: 853.3014085
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014085, upper bound: 853.3014085
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014181, upper bound: 853.3014732
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014181, upper bound: 853.3014181
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014330, upper bound: 853.3015014
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014528, upper bound: 853.3014661
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014204, upper bound: 853.3014259
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3014204, upper bound: 853.3014209
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005732
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005546
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3004503, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3004503, upper bound: 853.3004461
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3019116, upper bound: 853.3018669
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3018669, upper bound: 853.3018669
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3019386, upper bound: 853.3018699
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3018699, upper bound: 853.3018699
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3011919, upper bound: 853.3012255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3011919, upper bound: 853.3012255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3012421, upper bound: 853.3012009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3012421, upper bound: 853.3012009
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3011216, upper bound: 853.3011216
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3013073, upper bound: 853.3011216
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3012015, upper bound: 853.3010769
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3010874, upper bound: 853.3010769
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3012437, upper bound: 853.3012303
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.3012303, upper bound: 853.3012303
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2959777, upper bound: 853.2959660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2959919, upper bound: 853.2959660
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2916992, upper bound: 853.2916993
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2916992, upper bound: 853.2916993
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2915164, upper bound: 853.2914871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2915286, upper bound: 853.2914871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2958365, upper bound: 853.2958365
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.36
Output dim: 0, lower bound: -853.2958365, upper bound: 853.2958365

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012624, upper bound: 853.3013132
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012528, upper bound: 853.3012528
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014058, upper bound: 853.3014057
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014058, upper bound: 853.3014897
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013518, upper bound: 853.3013518
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013707, upper bound: 853.3013518
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004474
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013605, upper bound: 853.3013605
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013605, upper bound: 853.3013605
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008936, upper bound: 853.3010363
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008617, upper bound: 853.3010363
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014528, upper bound: 853.3014330
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014330, upper bound: 853.3014661
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014084, upper bound: 853.3014103
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014084, upper bound: 853.3014115
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014014, upper bound: 853.3014014
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014014, upper bound: 853.3014024
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005484
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005732
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005071, upper bound: 853.3005148
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005071, upper bound: 853.3005273
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004033, upper bound: 853.3003902
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003902, upper bound: 853.3003902
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018934, upper bound: 853.3018598
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019017, upper bound: 853.3018598
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008223, upper bound: 853.3008223
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008223, upper bound: 853.3008223
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017902, upper bound: 853.3017902
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018595, upper bound: 853.3017902
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011016, upper bound: 853.3011016
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011016, upper bound: 853.3011016
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011860, upper bound: 853.3012255
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011860, upper bound: 853.3011882
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011802, upper bound: 853.3012138
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011802, upper bound: 853.3011802
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006021, upper bound: 853.3006345
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006021, upper bound: 853.3006345
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011434, upper bound: 853.3011524
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012409, upper bound: 853.3011509
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010767, upper bound: 853.3010767
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010767, upper bound: 853.3010767
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3010782, upper bound: 853.3010782
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011205, upper bound: 853.3010782
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012040, upper bound: 853.3012040
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012187, upper bound: 853.3012040
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000224, upper bound: 853.3000224
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000224, upper bound: 853.3000224
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2805920, upper bound: 853.2805920
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2805920, upper bound: 853.2805920
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946552, upper bound: 853.2946552
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2946598, upper bound: 853.2946552
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2540701, upper bound: 853.2540701
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -853.2540701, upper bound: 853.2540701
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2951856, upper bound: 853.2951856
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2951856, upper bound: 853.2951856
time: 0.85 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3012624, upper bound: 853.3013132
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3012528, upper bound: 853.3012528
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014058, upper bound: 853.3014057
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014058, upper bound: 853.3014897
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3013518, upper bound: 853.3013518
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3013707, upper bound: 853.3013518
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004474
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3013605, upper bound: 853.3013605
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3013605, upper bound: 853.3013605
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3008936, upper bound: 853.3010363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3008617, upper bound: 853.3010363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014528, upper bound: 853.3014330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014330, upper bound: 853.3014661
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014084, upper bound: 853.3014103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014084, upper bound: 853.3014115
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014014, upper bound: 853.3014014
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3014014, upper bound: 853.3014024
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005484
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005732
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3005071, upper bound: 853.3005148
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3005071, upper bound: 853.3005273
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004033, upper bound: 853.3003902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3003902, upper bound: 853.3003902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3018934, upper bound: 853.3018598
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3019017, upper bound: 853.3018598
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3008223, upper bound: 853.3008223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3008223, upper bound: 853.3008223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3017902, upper bound: 853.3017902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3018595, upper bound: 853.3017902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011016, upper bound: 853.3011016
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011016, upper bound: 853.3011016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011860, upper bound: 853.3012255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011860, upper bound: 853.3011882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011802, upper bound: 853.3012138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011802, upper bound: 853.3011802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3006021, upper bound: 853.3006345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3006021, upper bound: 853.3006345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011434, upper bound: 853.3011524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3012409, upper bound: 853.3011509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3010767, upper bound: 853.3010767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3010767, upper bound: 853.3010767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3010782, upper bound: 853.3010782
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3011205, upper bound: 853.3010782
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3012040, upper bound: 853.3012040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3012187, upper bound: 853.3012040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3000224, upper bound: 853.3000224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.3000224, upper bound: 853.3000224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2805920, upper bound: 853.2805920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2805920, upper bound: 853.2805920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2946552, upper bound: 853.2946552
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2946598, upper bound: 853.2946552
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2540701, upper bound: 853.2540701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2540701, upper bound: 853.2540701
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2951856, upper bound: 853.2951856
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 0, lower bound: -853.2951856, upper bound: 853.2951856

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012594, upper bound: 853.3013099
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012595, upper bound: 853.3012501
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012528, upper bound: 853.3012528
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012528, upper bound: 853.3012528
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008421, upper bound: 853.3008421
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008421, upper bound: 853.3008421
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003732, upper bound: 853.3003732
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003732, upper bound: 853.3003732
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003391, upper bound: 853.3003391
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003391, upper bound: 853.3003391
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004979, upper bound: 853.3004830
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004904, upper bound: 853.3004830
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997499, upper bound: 853.2997499
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997499, upper bound: 853.2997499
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005010, upper bound: 853.3005017
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005010, upper bound: 853.3005010
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013127, upper bound: 853.3013127
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013127, upper bound: 853.3013127
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000397, upper bound: 853.3000397
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000397, upper bound: 853.3000397
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008562, upper bound: 853.3008360
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008595, upper bound: 853.3010038
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008610, upper bound: 853.3008610
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008936, upper bound: 853.3010319
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008789, upper bound: 853.3008789
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008804, upper bound: 853.3008789
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008617, upper bound: 853.3008617
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008617, upper bound: 853.3008617
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012483, upper bound: 853.3012406
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012552, upper bound: 853.3012392
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005307, upper bound: 853.3005307
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005307, upper bound: 853.3005307
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003360, upper bound: 853.3003360
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003360, upper bound: 853.3003360
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005484
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005484
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004066, upper bound: 853.3004087
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004066, upper bound: 853.3004078
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004582, upper bound: 853.3004636
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004582, upper bound: 853.3004582
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003507, upper bound: 853.3003573
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003507, upper bound: 853.3003545
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002605, upper bound: 853.3002605
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002681, upper bound: 853.3002605
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003902, upper bound: 853.3003902
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003902, upper bound: 853.3003902
time: 0.81 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3012594, upper bound: 853.3013099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3012595, upper bound: 853.3012501
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3012528, upper bound: 853.3012528
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3012528, upper bound: 853.3012528
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008421, upper bound: 853.3008421
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008421, upper bound: 853.3008421
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003732, upper bound: 853.3003732
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003732, upper bound: 853.3003732
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003391, upper bound: 853.3003391
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003391, upper bound: 853.3003391
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004979, upper bound: 853.3004830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004904, upper bound: 853.3004830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3014009, upper bound: 853.3014009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.2997499, upper bound: 853.2997499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.2997499, upper bound: 853.2997499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3005010, upper bound: 853.3005017
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3005010, upper bound: 853.3005010
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3013127, upper bound: 853.3013127
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3013127, upper bound: 853.3013127
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004531, upper bound: 853.3004531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3001512, upper bound: 853.3001512
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3000397, upper bound: 853.3000397
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3000397, upper bound: 853.3000397
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008562, upper bound: 853.3008360
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008595, upper bound: 853.3010038
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008610, upper bound: 853.3008610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008936, upper bound: 853.3010319
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008789, upper bound: 853.3008789
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008804, upper bound: 853.3008789
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008617, upper bound: 853.3008617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3008617, upper bound: 853.3008617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3012483, upper bound: 853.3012406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3012552, upper bound: 853.3012392
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3005307, upper bound: 853.3005307
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3005307, upper bound: 853.3005307
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003360, upper bound: 853.3003360
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003360, upper bound: 853.3003360
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005484
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3005484, upper bound: 853.3005484
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004066, upper bound: 853.3004087
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004066, upper bound: 853.3004078
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004582, upper bound: 853.3004636
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3004582, upper bound: 853.3004582
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003507, upper bound: 853.3003573
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003507, upper bound: 853.3003545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3002605, upper bound: 853.3002605
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3002681, upper bound: 853.3002605
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003902, upper bound: 853.3003902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.74
Output dim: 0, lower bound: -853.3003902, upper bound: 853.3003902
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3004461, upper bound: 853.3004461
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3018934, upper bound: 853.3018598
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3019017, upper bound: 853.3018598
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3008223, upper bound: 853.3008223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3008223, upper bound: 853.3008223
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3017902, upper bound: 853.3017902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3018595, upper bound: 853.3017902
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011016, upper bound: 853.3011016
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011016, upper bound: 853.3011016
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011860, upper bound: 853.3012255
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011860, upper bound: 853.3011882
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011802, upper bound: 853.3012138
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011802, upper bound: 853.3011802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3006021, upper bound: 853.3006345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3006021, upper bound: 853.3006345
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011434, upper bound: 853.3011524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3012409, upper bound: 853.3011509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3010767, upper bound: 853.3010767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3010767, upper bound: 853.3010767
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3010782, upper bound: 853.3010782
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3011205, upper bound: 853.3010782
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2994787, upper bound: 853.2994787
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3012040, upper bound: 853.3012040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3012187, upper bound: 853.3012040
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3000224, upper bound: 853.3000224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.3000224, upper bound: 853.3000224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2996485, upper bound: 853.2996485
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2946552, upper bound: 853.2946552
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2946598, upper bound: 853.2946552
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2951856, upper bound: 853.2951856
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.74
Output dim: 0, lower bound: -853.2951856, upper bound: 853.2951856
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3031061, upper bound: 853.3031189
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031061
time: 1.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.88 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -853.3031061, upper bound: 853.3031189
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.88
Output dim: 0, lower bound: -853.3031189, upper bound: 853.3031061

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3029483
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3029458, upper bound: 853.3028090
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030772, upper bound: 853.3031061
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3030772, upper bound: 853.3030792
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.49 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3029483
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -853.3029458, upper bound: 853.3028090
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -853.3030772, upper bound: 853.3031061
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -853.3030772, upper bound: 853.3030792

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3029483
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3029483
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027880, upper bound: 853.3028020
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028566, upper bound: 853.3027880
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3029483
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3029483
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3027880, upper bound: 853.3028020
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3028566, upper bound: 853.3027880
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028366, upper bound: 853.3027620
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028324, upper bound: 853.3028919
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3028672
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027983, upper bound: 853.3029483
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023811, upper bound: 853.3023468
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3023250, upper bound: 853.3023498
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014359, upper bound: 853.3014388
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014721, upper bound: 853.3014388
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001016
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001023
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000206, upper bound: 853.3000206
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000206, upper bound: 853.3000217
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001341
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000777
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000784
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3028366, upper bound: 853.3027620
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3028324, upper bound: 853.3028919
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3028438, upper bound: 853.3028672
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3027983, upper bound: 853.3029483
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3023811, upper bound: 853.3023468
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3023250, upper bound: 853.3023498
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3014359, upper bound: 853.3014388
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3014721, upper bound: 853.3014388
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001016
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001023
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3000206, upper bound: 853.3000206
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3000206, upper bound: 853.3000217
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001341
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000777
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000784

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027077, upper bound: 853.3025769
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027077, upper bound: 853.3025768
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028018, upper bound: 853.3028133
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027944, upper bound: 853.3027024
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028366, upper bound: 853.3028461
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028239, upper bound: 853.3028526
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028054, upper bound: 853.3029473
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3028077, upper bound: 853.3029473
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022515, upper bound: 853.3022314
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022927, upper bound: 853.3022508
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008231, upper bound: 853.3008231
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008231, upper bound: 853.3008231
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014300, upper bound: 853.3014388
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015482, upper bound: 853.3014300
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008125, upper bound: 853.3007669
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3008125, upper bound: 853.3007669
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001016
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001016
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000249
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999639
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001341
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000438
time: 0.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3027077, upper bound: 853.3025769
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3027077, upper bound: 853.3025768
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3028018, upper bound: 853.3028133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3027944, upper bound: 853.3027024
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3028366, upper bound: 853.3028461
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3028239, upper bound: 853.3028526
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3028054, upper bound: 853.3029473
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3028077, upper bound: 853.3029473
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3022515, upper bound: 853.3022314
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3022927, upper bound: 853.3022508
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3008231, upper bound: 853.3008231
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3008231, upper bound: 853.3008231
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3014300, upper bound: 853.3014388
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3015482, upper bound: 853.3014300
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3008125, upper bound: 853.3007669
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3008125, upper bound: 853.3007669
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001016
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3001016, upper bound: 853.3001016
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000249
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999639
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001341
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.78
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000438

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026857, upper bound: 853.3025769
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026412, upper bound: 853.3025768
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025428, upper bound: 853.3025082
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025437, upper bound: 853.3025082
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027221, upper bound: 853.3027567
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027308, upper bound: 853.3027211
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027337, upper bound: 853.3026944
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027785, upper bound: 853.3026916
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027396, upper bound: 853.3027222
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026749, upper bound: 853.3027222
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027589, upper bound: 853.3027610
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026258, upper bound: 853.3027997
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027866, upper bound: 853.3029407
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027975, upper bound: 853.3027672
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027603, upper bound: 853.3026881
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027509, upper bound: 853.3028499
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022140, upper bound: 853.3022140
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022207, upper bound: 853.3022217
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012963, upper bound: 853.3011989
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012046, upper bound: 853.3011989
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007804, upper bound: 853.3007804
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007804, upper bound: 853.3007804
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999855, upper bound: 853.2999855
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999855, upper bound: 853.2999855
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013092, upper bound: 853.3013179
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013092, upper bound: 853.3013092
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015482, upper bound: 853.3014300
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015185, upper bound: 853.3014300
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007479, upper bound: 853.3007331
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007331, upper bound: 853.3007331
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007125, upper bound: 853.3007125
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007305, upper bound: 853.3007125
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997578, upper bound: 853.2997578
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997578, upper bound: 853.2997578
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000215
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995620
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999294, upper bound: 853.2999294
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999294, upper bound: 853.2999294
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999018, upper bound: 853.2999018
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999018, upper bound: 853.2999027
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000777
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000777
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001341
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999763, upper bound: 853.2999763
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999763, upper bound: 853.2999763
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999357, upper bound: 853.2999357
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999357, upper bound: 853.2999357
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000438
time: 0.71 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3026857, upper bound: 853.3025769
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3026412, upper bound: 853.3025768
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3025428, upper bound: 853.3025082
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3025437, upper bound: 853.3025082
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027221, upper bound: 853.3027567
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027308, upper bound: 853.3027211
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027337, upper bound: 853.3026944
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027785, upper bound: 853.3026916
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027396, upper bound: 853.3027222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3026749, upper bound: 853.3027222
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027589, upper bound: 853.3027610
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3026258, upper bound: 853.3027997
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027866, upper bound: 853.3029407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027975, upper bound: 853.3027672
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027603, upper bound: 853.3026881
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3027509, upper bound: 853.3028499
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3022140, upper bound: 853.3022140
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3022207, upper bound: 853.3022217
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3012963, upper bound: 853.3011989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3012046, upper bound: 853.3011989
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3007804, upper bound: 853.3007804
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3007804, upper bound: 853.3007804
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999855, upper bound: 853.2999855
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999855, upper bound: 853.2999855
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3013092, upper bound: 853.3013179
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3013092, upper bound: 853.3013092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3015482, upper bound: 853.3014300
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3015185, upper bound: 853.3014300
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3007479, upper bound: 853.3007331
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3007331, upper bound: 853.3007331
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3007125, upper bound: 853.3007125
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3007305, upper bound: 853.3007125
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2997578, upper bound: 853.2997578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2997578, upper bound: 853.2997578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000215
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995620
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999294, upper bound: 853.2999294
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999294, upper bound: 853.2999294
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999018, upper bound: 853.2999018
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999018, upper bound: 853.2999027
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000777
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000777, upper bound: 853.3000777
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001350
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3001341, upper bound: 853.3001341
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999763, upper bound: 853.2999763
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999763, upper bound: 853.2999763
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999357, upper bound: 853.2999357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.2999357, upper bound: 853.2999357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -853.3000430, upper bound: 853.3000438

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026857, upper bound: 853.3025769
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025768, upper bound: 853.3025768
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027006, upper bound: 853.3025768
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027005, upper bound: 853.3025768
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026467, upper bound: 853.3025082
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026179, upper bound: 853.3025082
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025080, upper bound: 853.3024188
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3024978, upper bound: 853.3024188
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025583, upper bound: 853.3027118
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026731, upper bound: 853.3026301
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027201, upper bound: 853.3027211
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027308, upper bound: 853.3026105
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027091, upper bound: 853.3026269
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026579, upper bound: 853.3026040
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025938, upper bound: 853.3025087
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025835, upper bound: 853.3025087
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025569, upper bound: 853.3025461
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3025082, upper bound: 853.3025727
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026083, upper bound: 853.3026083
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026083, upper bound: 853.3026759
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027508, upper bound: 853.3027497
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027506, upper bound: 853.3027040
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026290, upper bound: 853.3027997
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3026258, upper bound: 853.3027741
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027255, upper bound: 853.3029142
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027260, upper bound: 853.3026973
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027308, upper bound: 853.3027305
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027576, upper bound: 853.3027305
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027508, upper bound: 853.3026790
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027516, upper bound: 853.3026790
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027437, upper bound: 853.3028434
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3027055, upper bound: 853.3026790
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017175, upper bound: 853.3017175
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017175, upper bound: 853.3017175
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022207, upper bound: 853.3022172
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3022140, upper bound: 853.3022217
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011182, upper bound: 853.3011182
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011207, upper bound: 853.3011182
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006490, upper bound: 853.3006490
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006490, upper bound: 853.3006490
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005834, upper bound: 853.3005834
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005834, upper bound: 853.3005834
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007497, upper bound: 853.3007497
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007497, upper bound: 853.3007497
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997198, upper bound: 853.2997198
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2997198, upper bound: 853.2997198
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999855, upper bound: 853.2999855
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999855, upper bound: 853.2999855
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007097, upper bound: 853.3007133
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007097, upper bound: 853.3007133
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006042, upper bound: 853.3006042
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006042, upper bound: 853.3006042
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007817, upper bound: 853.3006444
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006444, upper bound: 853.3006443
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007097, upper bound: 853.3007097
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3007097, upper bound: 853.3007097
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005790, upper bound: 853.3005790
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3005790, upper bound: 853.3005790
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006949, upper bound: 853.3006949
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3006949, upper bound: 853.3006949
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998274, upper bound: 853.2998274
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998274, upper bound: 853.2998274
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003333, upper bound: 853.3003331
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003333, upper bound: 853.3003331
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996051, upper bound: 853.2996051
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996051, upper bound: 853.2996051
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996667, upper bound: 853.2996667
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2996667, upper bound: 853.2996667
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000213
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000213, upper bound: 853.3000215
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995453, upper bound: 853.2995453
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995453, upper bound: 853.2995453
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995585, upper bound: 853.2995585
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991934, upper bound: 853.2991934
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2991934, upper bound: 853.2991934
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994747, upper bound: 853.2994783
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994747, upper bound: 853.2994747
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994891, upper bound: 853.2994891
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994891, upper bound: 853.2994891
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998666, upper bound: 853.2998666
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998666, upper bound: 853.2998666
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998165, upper bound: 853.2998165
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998165, upper bound: 853.2998165
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995540, upper bound: 853.2995540
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2995540, upper bound: 853.2995540
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994803, upper bound: 853.2994803
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2994803, upper bound: 853.2994803
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999630, upper bound: 853.2999630
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999018, upper bound: 853.2999018
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999018, upper bound: 853.2999018
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998038, upper bound: 853.2998038
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2998038, upper bound: 853.2998154
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000884, upper bound: 853.3000884
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.08 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=991.4861450195312
rel_dist={0: [-853.3031188934297, 853.3031188934297]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -853.3020713, upper bound: 853.3020713

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020057, upper bound: 853.3020057
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3020057, upper bound: 853.3020057
time: 0.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016274, upper bound: 853.3015901
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015901, upper bound: 853.3016274
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -853.3020057, upper bound: 853.3020057
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -853.3020057, upper bound: 853.3020057
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -853.3016274, upper bound: 853.3015901
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 0, lower bound: -853.3015901, upper bound: 853.3016274

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019475, upper bound: 853.3020057
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019475, upper bound: 853.3019480
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019425, upper bound: 853.3019887
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019869, upper bound: 853.3018975
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015755, upper bound: 853.3015549
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015994, upper bound: 853.3015500
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015484, upper bound: 853.3015852
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014980, upper bound: 853.3015827
time: 1.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3019475, upper bound: 853.3020057
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3019475, upper bound: 853.3019480
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3019425, upper bound: 853.3019887
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3019869, upper bound: 853.3018975
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3015755, upper bound: 853.3015549
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3015994, upper bound: 853.3015500
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3015484, upper bound: 853.3015852
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -853.3014980, upper bound: 853.3015827

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018466, upper bound: 853.3019113
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018460, upper bound: 853.3018554
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018942, upper bound: 853.3018878
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018942, upper bound: 853.3019480
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017750, upper bound: 853.3019423
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018322, upper bound: 853.3018463
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018452, upper bound: 853.3018488
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019433, upper bound: 853.3018157
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015129, upper bound: 853.3015549
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015755, upper bound: 853.3015223
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015895, upper bound: 853.3015239
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015994, upper bound: 853.3015500
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013061, upper bound: 853.3013298
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013061, upper bound: 853.3013298
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002034, upper bound: 853.3002328
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002179, upper bound: 853.3002328
time: 0.77 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3018466, upper bound: 853.3019113
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3018460, upper bound: 853.3018554
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3018942, upper bound: 853.3018878
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3018942, upper bound: 853.3019480
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3017750, upper bound: 853.3019423
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3018322, upper bound: 853.3018463
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3018452, upper bound: 853.3018488
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3019433, upper bound: 853.3018157
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3015129, upper bound: 853.3015549
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3015755, upper bound: 853.3015223
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3015895, upper bound: 853.3015239
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3015994, upper bound: 853.3015500
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3013061, upper bound: 853.3013298
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3013061, upper bound: 853.3013298
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3002034, upper bound: 853.3002328
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.90
Output dim: 0, lower bound: -853.3002179, upper bound: 853.3002328

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018361, upper bound: 853.3018314
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018253, upper bound: 853.3018864
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018398, upper bound: 853.3018554
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018398, upper bound: 853.3017722
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000797, upper bound: 853.3000766
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000766, upper bound: 853.3000766
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3019456, upper bound: 853.3019029
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018507, upper bound: 853.3019033
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017758, upper bound: 853.3018736
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017758, upper bound: 853.3019423
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017922, upper bound: 853.3018017
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018146, upper bound: 853.3018006
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017722, upper bound: 853.3017292
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017472, upper bound: 853.3017596
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017622, upper bound: 853.3017798
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018698, upper bound: 853.3017776
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015129, upper bound: 853.3015549
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015129, upper bound: 853.3015498
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014775, upper bound: 853.3014051
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014775, upper bound: 853.3014058
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015687, upper bound: 853.3014693
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015252, upper bound: 853.3014761
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015933, upper bound: 853.3015500
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015994, upper bound: 853.3015481
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012988, upper bound: 853.3013212
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012752, upper bound: 853.3013140
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002326, upper bound: 853.3002025
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3002326, upper bound: 853.3002025
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000040, upper bound: 853.3000029
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000316
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000185, upper bound: 853.3000075
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000316
time: 0.72 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018361, upper bound: 853.3018314
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018253, upper bound: 853.3018864
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018398, upper bound: 853.3018554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018398, upper bound: 853.3017722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3000797, upper bound: 853.3000766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3000766, upper bound: 853.3000766
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3019456, upper bound: 853.3019029
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018507, upper bound: 853.3019033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3017758, upper bound: 853.3018736
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3017758, upper bound: 853.3019423
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3017922, upper bound: 853.3018017
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018146, upper bound: 853.3018006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3017722, upper bound: 853.3017292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3017472, upper bound: 853.3017596
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3017622, upper bound: 853.3017798
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3018698, upper bound: 853.3017776
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3015129, upper bound: 853.3015549
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3015129, upper bound: 853.3015498
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3014775, upper bound: 853.3014051
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3014775, upper bound: 853.3014058
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3015687, upper bound: 853.3014693
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3015252, upper bound: 853.3014761
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3015933, upper bound: 853.3015500
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3015994, upper bound: 853.3015481
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3012988, upper bound: 853.3013212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3012752, upper bound: 853.3013140
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3002326, upper bound: 853.3002025
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3002326, upper bound: 853.3002025
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3000040, upper bound: 853.3000029
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000316
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3000185, upper bound: 853.3000075
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.37
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000316

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001022, upper bound: 853.3001022
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001023, upper bound: 853.3001022
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016950, upper bound: 853.3018244
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017829, upper bound: 853.3018469
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017812, upper bound: 853.3017949
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017812, upper bound: 853.3018554
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017267, upper bound: 853.3017267
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017267, upper bound: 853.3017267
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000797, upper bound: 853.3000766
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000766, upper bound: 853.3000766
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000242, upper bound: 853.3000242
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000271, upper bound: 853.3000242
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018242, upper bound: 853.3018944
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018242, upper bound: 853.3018944
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3018481, upper bound: 853.3017997
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017182, upper bound: 853.3017989
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016097, upper bound: 853.3016115
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015930, upper bound: 853.3016115
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017239, upper bound: 853.3018993
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017247, upper bound: 853.3018000
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003158, upper bound: 853.3003158
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3003158, upper bound: 853.3003158
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017240, upper bound: 853.3017919
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017240, upper bound: 853.3017856
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017719, upper bound: 853.3017292
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017292
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017596
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017326
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017149, upper bound: 853.3017149
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017149, upper bound: 853.3017363
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014625, upper bound: 853.3014554
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014555, upper bound: 853.3014554
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014596, upper bound: 853.3015146
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014606, upper bound: 853.3015149
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014051, upper bound: 853.3014058
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014051, upper bound: 853.3014152
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014083, upper bound: 853.3013343
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3013524, upper bound: 853.3013343
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011582, upper bound: 853.3011582
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3011582, upper bound: 853.3011582
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015236, upper bound: 853.3014267
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015237, upper bound: 853.3014152
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012230, upper bound: 853.3011933
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012230, upper bound: 853.3011939
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014858, upper bound: 853.3015214
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3015719, upper bound: 853.3014858
time: 1.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001413, upper bound: 853.3001288
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001413, upper bound: 853.3001288
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004037, upper bound: 853.3004037
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3004037, upper bound: 853.3004037
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012588, upper bound: 853.3012973
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3012644, upper bound: 853.3012588
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000626, upper bound: 853.3000626
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000626, upper bound: 853.3000626
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001995, upper bound: 853.3001653
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001681, upper bound: 853.3001653
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999462, upper bound: 853.2999539
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999462, upper bound: 853.2999462
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999992
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000075
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000185, upper bound: 853.3000048
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999529, upper bound: 853.2999823
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.2999529, upper bound: 853.2999529
time: 0.74 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3001022, upper bound: 853.3001022
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3001023, upper bound: 853.3001022
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3016950, upper bound: 853.3018244
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017829, upper bound: 853.3018469
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017812, upper bound: 853.3017949
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017812, upper bound: 853.3018554
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017267, upper bound: 853.3017267
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017267, upper bound: 853.3017267
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000797, upper bound: 853.3000766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000766, upper bound: 853.3000766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000242, upper bound: 853.3000242
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000271, upper bound: 853.3000242
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3018242, upper bound: 853.3018944
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3018242, upper bound: 853.3018944
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3018481, upper bound: 853.3017997
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017182, upper bound: 853.3017989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3016097, upper bound: 853.3016115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3015930, upper bound: 853.3016115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017239, upper bound: 853.3018993
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017247, upper bound: 853.3018000
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3003158, upper bound: 853.3003158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3003158, upper bound: 853.3003158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017240, upper bound: 853.3017919
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017240, upper bound: 853.3017856
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017719, upper bound: 853.3017292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017596
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017326
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017149, upper bound: 853.3017149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3017149, upper bound: 853.3017363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014625, upper bound: 853.3014554
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014555, upper bound: 853.3014554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014596, upper bound: 853.3015146
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014606, upper bound: 853.3015149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014051, upper bound: 853.3014058
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014051, upper bound: 853.3014152
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014083, upper bound: 853.3013343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3013524, upper bound: 853.3013343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3011582, upper bound: 853.3011582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3011582, upper bound: 853.3011582
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3015236, upper bound: 853.3014267
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3015237, upper bound: 853.3014152
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3012230, upper bound: 853.3011933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3012230, upper bound: 853.3011939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3014858, upper bound: 853.3015214
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3015719, upper bound: 853.3014858
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3001413, upper bound: 853.3001288
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3001413, upper bound: 853.3001288
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3004037, upper bound: 853.3004037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3004037, upper bound: 853.3004037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3012588, upper bound: 853.3012973
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3012644, upper bound: 853.3012588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000626, upper bound: 853.3000626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000626, upper bound: 853.3000626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3001995, upper bound: 853.3001653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3001681, upper bound: 853.3001653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.2999462, upper bound: 853.2999539
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.2999462, upper bound: 853.2999462
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999992
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000075
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.3000185, upper bound: 853.3000048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.2999529, upper bound: 853.2999823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.41
Output dim: 0, lower bound: -853.2999529, upper bound: 853.2999529

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001023, upper bound: 853.3001022
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3001022, upper bound: 853.3001022
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000770, upper bound: 853.3000770
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3000770, upper bound: 853.3000770
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014376, upper bound: 853.3015342
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014376, upper bound: 853.3015342
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016733, upper bound: 853.3018469
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017645, upper bound: 853.3017361
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014976, upper bound: 853.3014365
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014365, upper bound: 853.3014365
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017509, upper bound: 853.3018265
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017607, upper bound: 853.3017509
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014133, upper bound: 853.3014133
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3014133, upper bound: 853.3014133
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -182.7681427, 808.7180176, -182.7681427, 808.7180176, -991.4861450, 991.4861450
1: -228.5348969, 920.6278076, -228.5348969, 920.6278076, -1149.1625977, 1149.1625977
2: -233.1255188, 911.7817383, -233.1255188, 911.7817383, -1144.9072266, 1144.9072266
3: -369.3415222, 956.9161377, -369.3415222, 956.9161377, -1326.2575684, 1326.2575684
4: -372.4694519, 912.2335815, -372.4694519, 912.2335815, -1284.7027588, 1284.7027588

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3016857, upper bound: 853.3016857
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -853.3017518, upper bound: 853.3016857
time: 0.80 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3001023, upper bound: 853.3001022
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3001022, upper bound: 853.3001022
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3000770, upper bound: 853.3000770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3000770, upper bound: 853.3000770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3014376, upper bound: 853.3015342
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3014376, upper bound: 853.3015342
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3016733, upper bound: 853.3018469
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3017645, upper bound: 853.3017361
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3014976, upper bound: 853.3014365
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3014365, upper bound: 853.3014365
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3017509, upper bound: 853.3018265
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3017607, upper bound: 853.3017509
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3014133, upper bound: 853.3014133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3014133, upper bound: 853.3014133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3016857, upper bound: 853.3016857
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.19
Output dim: 0, lower bound: -853.3017518, upper bound: 853.3016857
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000797, upper bound: 853.3000766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000766, upper bound: 853.3000766
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000242, upper bound: 853.3000242
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000271, upper bound: 853.3000242
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3018242, upper bound: 853.3018944
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3018242, upper bound: 853.3018944
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3018481, upper bound: 853.3017997
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017182, upper bound: 853.3017989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3016097, upper bound: 853.3016115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3015930, upper bound: 853.3016115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017239, upper bound: 853.3018993
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017247, upper bound: 853.3018000
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3003158, upper bound: 853.3003158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3003158, upper bound: 853.3003158
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017240, upper bound: 853.3017919
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017240, upper bound: 853.3017856
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017719, upper bound: 853.3017292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017596
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017292, upper bound: 853.3017326
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017149, upper bound: 853.3017149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3017149, upper bound: 853.3017363
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014625, upper bound: 853.3014554
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014555, upper bound: 853.3014554
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014596, upper bound: 853.3015146
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014606, upper bound: 853.3015149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014051, upper bound: 853.3014058
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014051, upper bound: 853.3014152
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014083, upper bound: 853.3013343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3013524, upper bound: 853.3013343
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3011582, upper bound: 853.3011582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3011582, upper bound: 853.3011582
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3015236, upper bound: 853.3014267
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3015237, upper bound: 853.3014152
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3012230, upper bound: 853.3011933
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3012230, upper bound: 853.3011939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3014858, upper bound: 853.3015214
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3015719, upper bound: 853.3014858
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3001413, upper bound: 853.3001288
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3001413, upper bound: 853.3001288
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3004037, upper bound: 853.3004037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3004037, upper bound: 853.3004037
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3012588, upper bound: 853.3012973
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3012644, upper bound: 853.3012588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000626, upper bound: 853.3000626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000626, upper bound: 853.3000626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3001995, upper bound: 853.3001653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3001681, upper bound: 853.3001653
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.2999462, upper bound: 853.2999539
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.2999462, upper bound: 853.2999462
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999655
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.2999655, upper bound: 853.2999992
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000007, upper bound: 853.3000075
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.3000185, upper bound: 853.3000048
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.2999529, upper bound: 853.2999823
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.19
Output dim: 0, lower bound: -853.2999529, upper bound: 853.2999529
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=991.4861450195312
rel_dist={0: [-853.3029989672884, 853.3029989672882]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1105.74 seconds
