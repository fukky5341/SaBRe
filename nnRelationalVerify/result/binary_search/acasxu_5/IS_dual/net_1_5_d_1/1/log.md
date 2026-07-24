## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 560.5553892585241


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-152.4975433, 466.3875427, -152.4975433, 466.3875427, -618.8850708, 618.8850708)
1: (-216.3767242, 472.4903870, -216.3767242, 472.4903870, -688.8671265, 688.8671265)
2: (-182.8786926, 521.4256592, -182.8786926, 521.4256592, -704.3043213, 704.3042603)
3: (-194.7117004, 653.9572754, -194.7117004, 653.9572754, -848.6689453, 848.6689453)
4: (-163.1766510, 602.4576416, -163.1766510, 602.4576416, -765.6342773, 765.6342773)

## BASE Result
execution time: IAR + LP analysis = 1.72 + 2.25 = 3.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -560.5903858, upper bound: 560.5903858


# Binary Search by BASE starts (time budget: 1196.03 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=618.8850708007812
rel_dist={0: [-560.590385842507, 560.590385842507]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=618.8850708007812
rel_dist={0: [-560.5900507657058, 560.5900507657057]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=618.8850708007812
rel_dist={0: [-560.5891721066854, 560.5891721066855]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=618.8850708007812
rel_dist={0: [-560.588317335164, 560.588317335164]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=618.8850708007812
rel_dist={0: [-560.5876700627376, 560.5876700627373]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=618.8850708007812
rel_dist={0: [-560.5873221832865, 560.5873221832865]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=618.8850708007812
rel_dist={0: [-560.5871170365839, 560.587117036584]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=618.8850708007812
rel_dist={0: [-560.5869579945149, 560.5869579945149]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=618.8850708007812
rel_dist={0: [-560.586860278634, 560.586860278634]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=618.8850708007812
rel_dist={0: [-560.5868114206961, 560.586811420696]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=618.8850708007812
rel_dist={0: [-560.5867869917328, 560.5867869917327]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=618.8850708007812
rel_dist={0: [-560.5867747772616, 560.5867747772616]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=618.8850708007812
rel_dist={0: [-560.5867686700476, 560.5867686700476]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=618.8850708007812
rel_dist={0: [-560.5867656164455, 560.5867656164833]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=618.8850708007812
rel_dist={0: [-560.5867640897851, 560.586764089785]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=618.8850708007812
rel_dist={0: [-560.5867633246825, 560.5867633265998]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=618.8850708007812
rel_dist={0: [-560.5867629450911, 560.5867629450911]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=618.8850708007812
rel_dist={0: [-560.586762779438, 560.5867627533748]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=618.8850708007812
rel_dist={0: [-560.5867627256864, 560.586762675702]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=618.8850708007812
rel_dist={0: [-560.5867627285347, 560.5867626745653]}

## Binary Search Result
Binary search time: 77.44 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1118.59 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5903623
time: 0.95 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.96 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5903623
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -152.4975433, 466.3875427, -146.5755005, 447.3516541, -599.8491821, 612.9630127
1: -216.3767242, 472.4903870, -207.9424896, 453.5178833, -669.8945923, 680.4328613
2: -182.8786926, 521.4256592, -175.7731476, 500.5854492, -683.4640503, 697.1987915
3: -194.7117004, 653.9572754, -187.1182709, 627.4591675, -822.1708984, 841.0755005
4: -163.1766510, 602.4576416, -156.8196411, 578.2070312, -741.3836670, 759.2772217

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.83 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.72 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -151.8643036, 464.4644775, -188.6081696, 580.4994507, -732.3637085, 653.0726318
1: -215.4748840, 470.5242004, -267.9796143, 587.7565918, -803.2313843, 738.5037842
2: -182.1171265, 519.2548828, -226.6574402, 649.7827759, -831.8997803, 745.9122314
3: -193.8968964, 651.2348633, -241.4969330, 815.0511475, -1008.9479370, 892.7318115
4: -162.4932098, 599.9415283, -202.9143524, 751.9616089, -914.4547729, 802.8558350

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.93 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 1.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.57 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -146.5755005, 447.3516541, -593.9271240, 593.9270630
1: -207.9424896, 453.5178833, -207.9424896, 453.5178833, -661.4603882, 661.4603882
2: -175.7731476, 500.5854492, -175.7731476, 500.5854492, -676.3585815, 676.3585815
3: -187.1182709, 627.4591675, -187.1182709, 627.4591675, -814.5774536, 814.5774536
4: -156.8196411, 578.2070312, -156.8196411, 578.2070312, -735.0266724, 735.0266724

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
time: 1.21 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -146.5755005, 447.3516541, -635.9598389, 727.0748901
1: -267.9796143, 587.7565918, -207.9424896, 453.5178833, -721.4974976, 795.6990967
2: -226.6574402, 649.7827759, -175.7731476, 500.5854492, -727.2428589, 825.5559082
3: -241.4969330, 815.0511475, -187.1182709, 627.4591675, -868.9561157, 1002.1694336
4: -202.9143524, 751.9616089, -156.8196411, 578.2070312, -781.1213989, 908.7811890

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
time: 0.85 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -188.6081696, 580.4994507, -727.0748901, 635.9598389
1: -207.9424896, 453.5178833, -267.9796143, 587.7565918, -795.6990967, 721.4974976
2: -175.7731476, 500.5854492, -226.6574402, 649.7827759, -825.5559082, 727.2428589
3: -187.1182709, 627.4591675, -241.4969330, 815.0511475, -1002.1694336, 868.9561157
4: -156.8196411, 578.2070312, -202.9143524, 751.9616089, -908.7811890, 781.1213989

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5754267, upper bound: 560.5853710
time: 0.79 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.96 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -188.6081696, 580.4994507, -769.1076050, 769.1076050
1: -267.9796143, 587.7565918, -267.9796143, 587.7565918, -855.7362061, 855.7362061
2: -226.6574402, 649.7827759, -226.6574402, 649.7827759, -876.4401855, 876.4401855
3: -241.4969330, 815.0511475, -241.4969330, 815.0511475, -1056.5480957, 1056.5480957
4: -202.9143524, 751.9616089, -202.9143524, 751.9616089, -954.8759155, 954.8759155

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872007, upper bound: 560.5874456
time: 0.89 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872007, upper bound: 560.5881831
time: 0.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.94 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5754267, upper bound: 560.5853710
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5872007, upper bound: 560.5874456
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 5.94
Output dim: 0, lower bound: -560.5872007, upper bound: 560.5881831

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -142.9491425, 436.0071106, -582.5825806, 590.3007812
1: -207.9424896, 453.5178833, -202.7906494, 442.1823425, -650.1248169, 656.3085327
2: -175.7731476, 500.5854492, -171.3983307, 488.1363220, -663.9093628, 671.9837646
3: -187.1182709, 627.4591675, -182.4763184, 611.6849365, -798.8032227, 809.9354858
4: -156.8196411, 578.2070312, -152.9339752, 563.7099609, -720.5295410, 731.1409912

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896644
time: 0.83 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896971
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -146.3510437, 446.6672058, -154.1504974, 469.7902832, -616.1413574, 600.8176880
1: -207.6255493, 452.8211975, -218.4280548, 476.5923157, -684.2178345, 671.2492676
2: -175.5014801, 499.8098450, -184.8193207, 526.3300171, -701.8313599, 684.6291504
3: -186.8318787, 626.4881592, -196.7547760, 659.7593384, -846.5911865, 823.2429199
4: -156.5764465, 577.3077393, -164.9955444, 608.2030640, -764.7792969, 742.3032837

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896971, upper bound: 560.5896644
time: 1.23 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896971, upper bound: 560.5896971
time: 0.85 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -142.9491425, 436.0071106, -624.6152954, 723.4486084
1: -267.9796143, 587.7565918, -202.7906494, 442.1823425, -710.1619873, 790.5471802
2: -226.6574402, 649.7827759, -171.3983307, 488.1363220, -714.7937012, 821.1810913
3: -241.4969330, 815.0511475, -182.4763184, 611.6849365, -853.1818848, 997.5274658
4: -202.9143524, 751.9616089, -152.9339752, 563.7099609, -766.6242676, 904.8955688

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605809, upper bound: 560.5864791
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874128, upper bound: 560.5903623
time: 0.81 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
time: 0.89 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -188.3769989, 579.7989502, -154.1504974, 469.7902832, -658.1672974, 733.9494629
1: -267.6529541, 587.0421143, -218.4280548, 476.5923157, -744.2452393, 805.4700928
2: -226.3779755, 648.9873657, -184.8193207, 526.3300171, -752.7078857, 833.8067017
3: -241.2017059, 814.0588989, -196.7547760, 659.7593384, -900.9610596, 1010.8136597
4: -202.6641541, 751.0401001, -164.9955444, 608.2030640, -810.8671875, 916.0356445

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5606136, upper bound: 560.5857903
time: 0.88 seconds

## Relational analysis of IS_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5877388, upper bound: 560.5888447
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -128.8285217, 390.4894104, -188.6081696, 580.4994507, -709.3280029, 579.0975952
1: -182.5141296, 397.0623779, -267.9796143, 587.7565918, -770.2706299, 665.0419922
2: -154.2332153, 438.9310913, -226.6574402, 649.7827759, -804.0159302, 665.5884399
3: -164.3982239, 548.8854980, -241.4969330, 815.0511475, -979.4493408, 790.3824463
4: -137.6930389, 506.8053894, -202.9143524, 751.9616089, -889.6546631, 709.7197266

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864791, upper bound: 560.5598196
time: 0.83 seconds

## Relational analysis of IS_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5874128
time: 0.75 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812716, upper bound: 560.5858766
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -142.7839813, 435.4452515, -188.6081696, 580.4994507, -723.2834473, 624.0534058
1: -202.5985260, 441.5486145, -267.9796143, 587.7565918, -790.3550415, 709.5281982
2: -171.2445984, 487.3898621, -226.6574402, 649.7827759, -821.0273438, 714.0473022
3: -182.3119965, 610.7657471, -241.4969330, 815.0511475, -997.3630981, 852.2626343
4: -152.7651367, 562.8719482, -202.9143524, 751.9616089, -904.7267456, 765.7862549

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864791, upper bound: 560.5606136
time: 1.01 seconds

## Relational analysis of IS_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5881740
time: 0.87 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812716, upper bound: 560.5882067
time: 0.95 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -167.2992096, 512.3647461, -188.5758667, 580.3987427, -747.6979370, 700.9406128
1: -236.6211090, 518.4329834, -267.9344788, 587.6557007, -824.2767944, 786.3674316
2: -200.0903778, 572.9671021, -226.6191254, 649.6714478, -849.7618408, 799.5862427
3: -213.5314636, 718.2940063, -241.4563599, 814.9094849, -1028.4407959, 959.7503662
4: -179.1565399, 662.5183716, -202.8801575, 751.8314819, -930.9880371, 865.3985596

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5580392, upper bound: 560.5830079
time: 0.85 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528749, upper bound: 560.5552906
time: 0.83 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -184.3911133, 568.1159668, -188.6081696, 580.4994507, -764.8905029, 756.7241211
1: -261.9881287, 575.1311646, -267.9796143, 587.7565918, -849.7447510, 843.1107788
2: -221.6778412, 635.8009033, -226.6574402, 649.7827759, -871.4606323, 862.4583740
3: -236.1141052, 797.7171631, -241.4969330, 815.0511475, -1051.1652832, 1039.2141113
4: -198.4534302, 735.7338867, -202.9143524, 751.9616089, -950.4149780, 938.6482544

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5568049
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 1.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.62 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896644
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896971
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5896971, upper bound: 560.5896644
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5896971, upper bound: 560.5896971
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5874128, upper bound: 560.5903623
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5877388, upper bound: 560.5888447
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5874128
IS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5812716, upper bound: 560.5858766
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5881740
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5812716, upper bound: 560.5882067
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5580392, upper bound: 560.5830079
IS_B2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5528749, upper bound: 560.5552906
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5568049
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -142.9491425, 436.0071106, -142.9491425, 436.0071106, -578.9562378, 578.9562378
1: -202.7906494, 442.1823425, -202.7906494, 442.1823425, -644.9730225, 644.9730225
2: -171.3983307, 488.1363220, -171.3983307, 488.1363220, -659.5346680, 659.5346069
3: -182.4763184, 611.6849365, -182.4763184, 611.6849365, -794.1612549, 794.1612549
4: -152.9339752, 563.7099609, -152.9339752, 563.7099609, -716.6439209, 716.6439209

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5837290, upper bound: 560.5551012
time: 1.44 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5899180
time: 0.83 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5903532
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -142.9491425, 436.0071106, -590.1575928, 612.7394409
1: -218.4280548, 476.5923157, -202.7906494, 442.1823425, -660.6104126, 679.3828125
2: -184.8193207, 526.3300171, -171.3983307, 488.1363220, -672.9555054, 697.7283325
3: -196.7547760, 659.7593384, -182.4763184, 611.6849365, -808.4396973, 842.2356567
4: -164.9955444, 608.2030640, -152.9339752, 563.7099609, -728.7055054, 761.1369629

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
time: 0.92 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -142.9491425, 436.0071106, -154.1504974, 469.7902832, -612.7394409, 590.1575928
1: -202.7906494, 442.1823425, -218.4280548, 476.5923157, -679.3828125, 660.6104126
2: -171.3983307, 488.1363220, -184.8193207, 526.3300171, -697.7283325, 672.9555054
3: -182.4763184, 611.6849365, -196.7547760, 659.7593384, -842.2356567, 808.4396973
4: -152.9339752, 563.7099609, -164.9955444, 608.2030640, -761.1369629, 728.7055054

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812625, upper bound: 560.5873670
time: 1.17 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896644
time: 1.25 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -154.1504974, 469.7902832, -623.9407959, 623.9407959
1: -218.4280548, 476.5923157, -218.4280548, 476.5923157, -695.0202637, 695.0202637
2: -184.8193207, 526.3300171, -184.8193207, 526.3300171, -711.1492920, 711.1493530
3: -196.7547760, 659.7593384, -196.7547760, 659.7593384, -856.5140991, 856.5140991
4: -164.9955444, 608.2030640, -164.9955444, 608.2030640, -773.1986084, 773.1986084

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5892292, upper bound: 560.5888683
time: 0.87 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896971
time: 0.90 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -125.0910950, 378.7121887, -567.3203735, 705.5905762
1: -267.9796143, 587.7565918, -177.2164612, 385.2739868, -653.2536011, 764.9730225
2: -226.6574402, 649.7827759, -149.7499847, 425.9759521, -652.6334229, 799.5327148
3: -241.4969330, 815.0511475, -159.6127014, 532.4531860, -773.9501343, 974.6638184
4: -202.9143524, 751.9616089, -133.6832275, 491.7020264, -694.6163330, 885.6447144

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5598196, upper bound: 560.5864791
time: 0.99 seconds

## Relational analysis of IS_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874128, upper bound: 560.5898574
time: 1.05 seconds

## Relational analysis of IS_B1_A2_B1_B1_B2

### Relational analysis result of IS_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874128, upper bound: 560.5903623
time: 0.87 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -138.3768463, 421.7382202, -610.3463745, 718.8762207
1: -267.9796143, 587.7565918, -196.3459320, 427.8159790, -695.7955322, 784.1025391
2: -226.6574402, 649.7827759, -165.9172516, 472.2707825, -698.9281616, 815.7000122
3: -241.4969330, 815.0511475, -176.6769562, 591.6629028, -833.1598511, 991.7280884
4: -202.9143524, 751.9616089, -148.0329285, 545.2708130, -748.1851807, 899.9944458

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605809, upper bound: 560.5864791
time: 1.09 seconds

## Relational analysis of IS_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878870, upper bound: 560.5898574
time: 0.71 seconds

## Relational analysis of IS_B1_A2_B1_B2_B2

### Relational analysis result of IS_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -188.3446960, 579.6983643, -134.8042450, 412.6112061, -600.9558105, 714.5025024
1: -267.6078186, 586.9411011, -190.1982117, 417.1871033, -684.7948608, 777.1392822
2: -226.3396759, 648.8760376, -160.6754913, 460.2108765, -686.5504761, 809.5515137
3: -241.1611633, 813.9174194, -171.6001434, 578.0089722, -819.1701660, 985.5175781
4: -202.6299744, 750.9100342, -143.5054932, 531.5460205, -734.1759644, 894.4154663

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601208, upper bound: 560.5849615
time: 1.07 seconds

## Relational analysis of IS_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5881072
time: 0.98 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5888447
time: 0.86 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -188.3769989, 579.7989502, -151.5404358, 462.2336121, -650.6105957, 731.3393555
1: -267.6529541, 587.0421143, -214.9032593, 468.8790283, -736.5319824, 801.9453125
2: -226.3779755, 648.9873657, -181.8432465, 517.7068481, -744.0847778, 830.8305664
3: -241.2017059, 814.0588989, -193.5249939, 649.0505981, -890.2523193, 1007.5838623
4: -202.6641541, 751.0401001, -162.3022156, 598.2625732, -800.9267578, 913.3422852

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5606136, upper bound: 560.5857903
time: 0.86 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858766, upper bound: 560.5812716
time: 1.21 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -125.0910950, 378.7121887, -188.6081696, 580.4994507, -705.5905762, 567.3203735
1: -177.2164612, 385.2739868, -267.9796143, 587.7565918, -764.9730225, 653.2536011
2: -149.7499847, 425.9759521, -226.6574402, 649.7827759, -799.5327148, 652.6334229
3: -159.6127014, 532.4531860, -241.4969330, 815.0511475, -974.6638184, 773.9501343
4: -133.6832275, 491.7020264, -202.9143524, 751.9616089, -885.6447144, 694.6163330

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864791, upper bound: 560.5598196
time: 1.00 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5898574, upper bound: 560.5874128
time: 0.84 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5874128
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -132.6759491, 400.6557617, -188.3769989, 579.7989502, -712.4748535, 589.0327759
1: -187.9339905, 407.6807861, -267.6529541, 587.0421143, -774.9760742, 675.3336182
2: -159.0463715, 450.7799683, -226.3779755, 648.9873657, -808.0337524, 677.1579590
3: -169.1976929, 563.6649170, -241.2017059, 814.0588989, -983.2565918, 804.8666382
4: -141.8902435, 520.6527710, -202.6641541, 751.0401001, -892.9303589, 723.3168945

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773884, upper bound: 560.5582835
time: 0.98 seconds

## Relational analysis of IS_B2_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5805341, upper bound: 560.5848942
time: 0.93 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812716, upper bound: 560.5858766
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -188.6081696, 580.4994507, -718.8762207, 610.3463745
1: -196.3459320, 427.8159790, -267.9796143, 587.7565918, -784.1025391, 695.7954712
2: -165.9172516, 472.2707825, -226.6574402, 649.7827759, -815.7000122, 698.9281616
3: -176.6769562, 591.6629028, -241.4969330, 815.0511475, -991.7280884, 833.1598511
4: -148.0329285, 545.2708130, -202.9143524, 751.9616089, -899.9944458, 748.1851807

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864791, upper bound: 560.5605809
time: 0.76 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5898574, upper bound: 560.5878870
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5881740
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -149.9529419, 456.3767700, -188.3769989, 579.7989502, -729.7518921, 644.7537842
1: -212.5085449, 463.2142944, -267.6529541, 587.0421143, -799.5506592, 730.8672485
2: -179.8332214, 511.6408691, -226.3779755, 648.9873657, -828.8205566, 738.0187988
3: -191.4326782, 641.0862427, -241.2017059, 814.0588989, -1005.4915771, 882.2879639
4: -160.5236359, 591.1001587, -202.6641541, 751.0401001, -911.5637207, 793.7642822

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5857903, upper bound: 560.5606136
time: 0.97 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888447, upper bound: 560.5877388
time: 0.73 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896736, upper bound: 560.5882067
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -167.2992096, 512.3647461, -180.7823944, 556.0125732, -723.3117676, 693.1471558
1: -236.6211090, 518.4329834, -256.8913574, 563.1463623, -799.7674561, 775.3242798
2: -200.0903778, 572.9671021, -217.2644043, 622.7058716, -822.7962646, 790.2315063
3: -213.5314636, 718.2940063, -231.4853821, 780.7312622, -994.2626953, 949.7793579
4: -179.1565399, 662.5183716, -194.4803162, 720.3654175, -899.5219727, 856.9986572

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528639, upper bound: 560.5552906
time: 1.26 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528638, upper bound: 560.5552906
time: 0.85 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -176.0407715, 542.0167847, -188.6081696, 580.4994507, -756.5402222, 730.6249390
1: -250.2174072, 548.9858398, -267.9796143, 587.7565918, -837.9739990, 816.9654541
2: -211.7223206, 607.0142212, -226.6574402, 649.7827759, -861.5050659, 833.6716309
3: -225.4848785, 761.2257690, -241.4969330, 815.0511475, -1040.5360107, 1002.7227173
4: -189.5153198, 702.1432495, -202.9143524, 751.9616089, -941.4769287, 905.0575562

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 1.03 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563202, upper bound: 560.5567068
time: 0.81 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -274.8025513, 851.1666260, -187.2795258, 576.1981812, -851.0006104, 1036.2358398
1: -390.8899536, 860.7320557, -266.0727539, 583.4205933, -973.4581299, 1123.9528809
2: -329.9025574, 951.4403076, -225.0539398, 644.9916992, -974.2694092, 1172.8292236
3: -352.3719177, 1190.1116943, -239.7787933, 809.0115967, -1161.3834229, 1427.6604004
4: -295.7261047, 1096.5043945, -201.4813538, 746.4171143, -1041.4874268, 1296.6286621

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 1.02 seconds

## Relational analysis of IS_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.85 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5899180
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5903532
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5812625, upper bound: 560.5873670
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896644
IS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5892292, upper bound: 560.5888683
IS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5896644, upper bound: 560.5896971
IS_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5874128, upper bound: 560.5898574
IS_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5874128, upper bound: 560.5903623
IS_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5878870, upper bound: 560.5898574
IS_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5881740, upper bound: 560.5903623
IS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5881072
IS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5888447
IS_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5858766, upper bound: 560.5812716
IS_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5882067, upper bound: 560.5896736
IS_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5898574, upper bound: 560.5874128
IS_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5874128
IS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5805341, upper bound: 560.5848942
IS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5812716, upper bound: 560.5858766
IS_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5898574, upper bound: 560.5878870
IS_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5881740
IS_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5888447, upper bound: 560.5877388
IS_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5896736, upper bound: 560.5882067
IS_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5528639, upper bound: 560.5552906
IS_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5528638, upper bound: 560.5552906
IS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
IS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5563202, upper bound: 560.5567068
IS_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
IS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.85
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -126.4899063, 388.4345703, -142.9065094, 435.8724976, -562.3623657, 531.3410645
1: -178.4534149, 392.6055908, -202.7316284, 442.0486755, -620.5020142, 595.3371582
2: -150.6621704, 432.9809265, -171.3486328, 487.9895630, -638.6517334, 604.3295898
3: -160.9440765, 543.8226929, -182.4232941, 611.4984131, -772.4423828, 726.2458496
4: -134.6369324, 499.7395325, -152.8895721, 563.5388794, -698.1757812, 652.6290283

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5541903, upper bound: 560.5846086
time: 0.84 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5484864, upper bound: 560.5472007
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -140.2063599, 427.9375916, -142.9491425, 436.0071106, -576.2135010, 570.8867188
1: -199.0193481, 433.9468994, -202.7906494, 442.1823425, -641.2016602, 636.7374878
2: -168.2114410, 479.0105591, -171.3983307, 488.1363220, -656.3475342, 650.4088745
3: -179.0332031, 600.2573853, -182.4763184, 611.6849365, -790.7181396, 782.7337036
4: -150.0622406, 553.1212769, -152.9339752, 563.7099609, -713.7720947, 706.0552368

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5528245, upper bound: 560.5846086
time: 1.02 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5472007, upper bound: 560.5472007
time: 1.26 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -125.0910950, 378.7121887, -532.8626709, 594.8813477
1: -218.4280548, 476.5923157, -177.2164612, 385.2739868, -603.7020264, 653.8086548
2: -184.8193207, 526.3300171, -149.7499847, 425.9759521, -610.7952881, 676.0798950
3: -196.7547760, 659.7593384, -159.6127014, 532.4531860, -729.2079468, 819.3720093
4: -164.9955444, 608.2030640, -133.6832275, 491.7020264, -656.6975708, 741.8861084

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5898809
time: 1.25 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -138.3768463, 421.7382202, -575.8887329, 608.1671143
1: -218.4280548, 476.5923157, -196.3459320, 427.8159790, -646.2438965, 672.9381714
2: -184.8193207, 526.3300171, -165.9172516, 472.2707825, -657.0900879, 692.2471924
3: -196.7547760, 659.7593384, -176.6769562, 591.6629028, -788.4176636, 836.4362183
4: -164.9955444, 608.2030640, -148.0329285, 545.2708130, -710.2663574, 756.2358398

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5780020, upper bound: 560.5420016
time: 1.09 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893775, upper bound: 560.5898809
time: 0.95 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -125.0910950, 378.7121887, -154.1504974, 469.7902832, -594.8813477, 532.8626709
1: -177.2164612, 385.2739868, -218.4280548, 476.5923157, -653.8086548, 603.7020264
2: -149.7499847, 425.9759521, -184.8193207, 526.3300171, -676.0798950, 610.7952881
3: -159.6127014, 532.4531860, -196.7547760, 659.7593384, -819.3720093, 729.2079468
4: -133.6832275, 491.7020264, -164.9955444, 608.2030640, -741.8861084, 656.6975708

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5898809, upper bound: 560.5889032
time: 0.87 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903858, upper bound: 560.5889032
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -154.1504974, 469.7902832, -608.1670532, 575.8887329
1: -196.3459320, 427.8159790, -218.4280548, 476.5923157, -672.9381714, 646.2438965
2: -165.9172516, 472.2707825, -184.8193207, 526.3300171, -692.2471924, 657.0900879
3: -176.6769562, 591.6629028, -196.7547760, 659.7593384, -836.4362183, 788.4176636
4: -148.0329285, 545.2708130, -164.9955444, 608.2030640, -756.2358398, 710.2663574

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5420016, upper bound: 560.5780020
time: 0.98 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5898809, upper bound: 560.5893775
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903858, upper bound: 560.5896644
time: 0.84 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -154.1096191, 469.6587219, -134.8042450, 412.6112061, -566.7207031, 604.4628906
1: -218.3698120, 476.4615479, -190.1982117, 417.1871033, -635.5568848, 666.6597900
2: -184.7702942, 526.1864624, -160.6754913, 460.2108765, -644.9812012, 686.8619385
3: -196.7025299, 659.5767212, -171.6001434, 578.0089722, -774.7114868, 831.1768799
4: -164.9518280, 608.0355225, -143.5054932, 531.5460205, -696.4978638, 751.5409546

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884004, upper bound: 560.5884004
time: 0.81 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884004, upper bound: 560.5888683
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -151.5404358, 462.2336121, -616.3840942, 621.3306885
1: -218.4280548, 476.5923157, -214.9032593, 468.8790283, -687.3070679, 691.4954834
2: -184.8193207, 526.3300171, -181.8432465, 517.7068481, -702.5261841, 708.1731567
3: -196.7547760, 659.7593384, -193.5249939, 649.0505981, -845.8053589, 853.2843018
4: -164.9955444, 608.2030640, -162.3022156, 598.2625732, -763.2581177, 770.5052490

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5892292
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5896971
time: 0.86 seconds

## BFS IS instance: IS_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -186.4368896, 573.0894165, -107.9414673, 328.2743835, -514.7113037, 681.0308838
1: -264.8499146, 580.4411621, -151.8074799, 332.8845215, -597.7343140, 732.2485352
2: -224.0262756, 641.7292480, -128.1442413, 367.7981567, -591.8244629, 769.8734741
3: -238.6783142, 804.7745972, -137.1543121, 460.6455383, -699.3238525, 941.9288940
4: -200.5567932, 742.5912476, -114.6111145, 424.2887878, -624.8455200, 857.2022095

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5598196, upper bound: 560.5859741
time: 0.94 seconds

## Relational analysis of IS_B1_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B1_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5807427, upper bound: 560.5775078
time: 1.05 seconds

## Relational analysis of IS_B1_A2_B1_B1_B1_B2

### Relational analysis result of IS_B1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5816950, upper bound: 560.5836588
time: 0.94 seconds

## BFS IS instance: IS_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -122.3909149, 370.8732300, -559.4813843, 702.8903809
1: -267.9796143, 587.7565918, -173.5694885, 377.3158875, -645.2954712, 761.3259888
2: -226.6574402, 649.7827759, -146.6701202, 417.0385742, -643.6960449, 796.4528198
3: -241.4969330, 815.0511475, -156.2867889, 521.4085083, -762.9054565, 971.3379517
4: -202.9143524, 751.9616089, -130.9183655, 481.4752197, -684.3895874, 882.8799438

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5598196, upper bound: 560.5864791
time: 1.43 seconds

## Relational analysis of IS_B1_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5805618, upper bound: 560.5760622
time: 0.72 seconds

## Relational analysis of IS_B1_A2_B1_B1_B2_B2

### Relational analysis result of IS_B1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5816950, upper bound: 560.5844686
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -188.5758667, 580.3987427, -122.1466141, 374.9494019, -563.5252686, 702.5453491
1: -267.9344788, 587.6557007, -172.3555145, 379.0198364, -646.9543457, 760.0112305
2: -226.6191254, 649.6714478, -145.4720612, 417.9904480, -644.6095581, 795.1434326
3: -241.4563599, 814.9094849, -155.4417114, 524.9205933, -766.3769531, 970.3511353
4: -202.8801575, 751.8314819, -129.9905243, 482.3223877, -685.2025146, 881.8220215

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5602939, upper bound: 560.5859741
time: 0.77 seconds

## Relational analysis of IS_B1_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5765116, upper bound: 560.5410407
time: 0.98 seconds

## Relational analysis of IS_B1_A2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869046, upper bound: 560.5891198
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869046, upper bound: 560.5898574
time: 2.00 seconds

## BFS IS instance: IS_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -135.6968994, 413.9247437, -602.5328979, 716.1963501
1: -267.9796143, 587.7565918, -192.6562042, 419.8168030, -687.7963867, 780.4127808
2: -226.6574402, 649.7827759, -162.7923889, 463.3983154, -690.0556641, 812.5751343
3: -241.4969330, 815.0511475, -173.3052979, 580.5690308, -822.0659790, 988.3564453
4: -202.9143524, 751.9616089, -145.2133636, 534.9701538, -737.8845215, 897.1749878

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B2_B2_B1

### Relational analysis result of IS_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5598196, upper bound: 560.5864791
time: 1.07 seconds

## Relational analysis of IS_B1_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871916, upper bound: 560.5896248
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871916, upper bound: 560.5903623
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -167.0556488, 511.6388550, -134.8042450, 412.6112061, -579.6668091, 646.4429932
1: -236.2775574, 517.6907349, -190.1982117, 417.1871033, -653.4645996, 707.8889160
2: -199.7960968, 572.1392212, -160.6754913, 460.2108765, -660.0068359, 732.8146973
3: -213.2209320, 717.2656250, -171.6001434, 578.0089722, -791.2297974, 888.8657837
4: -178.8909302, 661.5563354, -143.5054932, 531.5460205, -710.4369507, 805.0617676

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5575700, upper bound: 560.5836695
time: 1.04 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799119, upper bound: 560.5876220
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5799119, upper bound: 560.5881072
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -184.1753693, 567.4651489, -134.8042450, 412.6112061, -596.7864990, 702.2694092
1: -261.6823730, 574.4641724, -190.1982117, 417.1871033, -678.8695068, 764.6623535
2: -221.4158020, 635.0573730, -160.6754913, 460.2108765, -681.6265869, 795.7328491
3: -235.8376923, 796.7936401, -171.6001434, 578.0089722, -813.8466187, 968.3937988
4: -198.2189484, 734.8729248, -143.5054932, 531.5460205, -729.7648926, 878.3783569

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5575700, upper bound: 560.5849615
time: 0.88 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5788402, upper bound: 560.5772482
time: 0.90 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5808445, upper bound: 560.5834463
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -188.3769989, 579.7989502, -130.2356262, 393.7623901, -582.1393433, 710.0345459
1: -267.6529541, 587.0421143, -184.7286530, 400.6760254, -668.3289185, 771.7707520
2: -226.3779755, 648.9873657, -156.3427277, 442.9280396, -669.3060303, 805.3300781
3: -241.2017059, 814.0588989, -166.2758484, 553.9575806, -795.1593018, 980.3347168
4: -202.6641541, 751.0401001, -139.4627686, 511.6647644, -714.3289185, 890.5028687

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5582835, upper bound: 560.5773884
time: 0.92 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5854025, upper bound: 560.5787427
time: 0.83 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5767921, upper bound: 560.5550214
time: 0.90 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858766, upper bound: 560.5811755
time: 0.84 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731201, upper bound: 560.5783716
time: 0.68 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731201, upper bound: 560.5812716
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -188.3769989, 579.7989502, -147.4514923, 449.1673584, -637.5443726, 727.2504272
1: -267.6529541, 587.0421143, -209.1342621, 455.8430176, -723.4959717, 796.1763306
2: -226.3779755, 648.9873657, -176.9795227, 503.3908386, -729.7687988, 825.9669189
3: -241.2017059, 814.0588989, -188.3370514, 630.8516235, -872.0533447, 1002.3958130
4: -202.6641541, 751.0401001, -157.9375305, 581.5914307, -784.2556152, 908.9776001

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5582835, upper bound: 560.5857903
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5889360
time: 1.08 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5896736
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -107.9414673, 328.2743835, -186.4368896, 573.0894165, -681.0308838, 514.7112427
1: -151.8074799, 332.8845215, -264.8499146, 580.4411621, -732.2485352, 597.7343750
2: -128.1442413, 367.7981567, -224.0262756, 641.7292480, -769.8734741, 591.8244629
3: -137.1543121, 460.6455383, -238.6783142, 804.7745972, -941.9288940, 699.3238525
4: -114.6111145, 424.2887878, -200.5567932, 742.5912476, -857.2022095, 624.8455200

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5859741, upper bound: 560.5598196
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5775078, upper bound: 560.5807427
time: 0.85 seconds

## Relational analysis of IS_B2_A1_A1_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836588, upper bound: 560.5816950
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -122.3909149, 370.8732300, -188.6081696, 580.4994507, -702.8903809, 559.4813843
1: -173.5694885, 377.3158875, -267.9796143, 587.7565918, -761.3259888, 645.2955322
2: -146.6701202, 417.0385742, -226.6574402, 649.7827759, -796.4528198, 643.6959839
3: -156.2867889, 521.4085083, -241.4969330, 815.0511475, -971.3379517, 762.9054565
4: -130.9183655, 481.4752197, -202.9143524, 751.9616089, -882.8799438, 684.3895874

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864791, upper bound: 560.5598196
time: 0.89 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A1_A2_A1

### Relational analysis result of IS_B2_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5760622, upper bound: 560.5805618
time: 1.00 seconds

## Relational analysis of IS_B2_A1_A1_A1_A2_A2

### Relational analysis result of IS_B2_A1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5844686, upper bound: 560.5816950
time: 0.85 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -132.6345825, 400.5218201, -167.0556488, 511.6388550, -644.2731934, 567.5773315
1: -187.8751221, 407.5474548, -236.2775574, 517.6907349, -705.5658569, 643.8249512
2: -158.9969635, 450.6340332, -199.7960968, 572.1392212, -731.1361694, 650.4300537
3: -169.1448822, 563.4789429, -213.2209320, 717.2656250, -886.4104004, 776.6997681
4: -141.8461914, 520.4822998, -178.8909302, 661.5563354, -803.4025269, 699.3731689

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5760964, upper bound: 560.5557326
time: 0.83 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5780051, upper bound: 560.5842816
time: 0.82 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5542391, upper bound: 560.5763269
time: 0.80 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5805341, upper bound: 560.5848942
time: 0.75 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5595502, upper bound: 560.5817515
time: 0.84 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5777630, upper bound: 560.5842566
time: 0.89 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -132.6759491, 400.6557617, -184.1753693, 567.4651489, -700.1410522, 584.8311157
1: -187.9339905, 407.6807861, -261.6823730, 574.4641724, -762.3981934, 669.3630981
2: -159.0463715, 450.7799683, -221.4158020, 635.0573730, -794.1037598, 672.1958008
3: -169.1976929, 563.6649170, -235.8376923, 796.7936401, -965.9912720, 799.5026245
4: -141.8902435, 520.6527710, -198.2189484, 734.8729248, -876.7631226, 718.8715820

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5773884, upper bound: 560.5582835
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5787427, upper bound: 560.5854025
time: 0.98 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5550214, upper bound: 560.5767921
time: 0.92 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5783716, upper bound: 560.5731201
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5783716, upper bound: 560.5858766
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -122.1466141, 374.9494019, -188.5758667, 580.3987427, -702.5453491, 563.5252686
1: -172.3555145, 379.0198364, -267.9344788, 587.6557007, -760.0112305, 646.9543457
2: -145.4720612, 417.9904480, -226.6191254, 649.6714478, -795.1434326, 644.6095581
3: -155.4417114, 524.9205933, -241.4563599, 814.9094849, -970.3511353, 766.3769531
4: -129.9905243, 482.3223877, -202.8801575, 751.8314819, -881.8220215, 685.2025146

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5859741, upper bound: 560.5602939
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5410407, upper bound: 560.5765116
time: 0.93 seconds

## Relational analysis of IS_B2_A1_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891198, upper bound: 560.5869046
time: 0.87 seconds

## Relational analysis of IS_B2_A1_A2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891198, upper bound: 560.5878870
time: 0.92 seconds

## BFS IS instance: IS_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -135.6968994, 413.9247437, -188.6081696, 580.4994507, -716.1963501, 602.5328979
1: -192.6562042, 419.8168030, -267.9796143, 587.7565918, -780.4127808, 687.7963867
2: -162.7923889, 463.3983154, -226.6574402, 649.7827759, -812.5751343, 690.0556641
3: -173.3052979, 580.5690308, -241.4969330, 815.0511475, -988.3564453, 822.0659790
4: -145.2133636, 534.9701538, -202.9143524, 751.9616089, -897.1749878, 737.8845215

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864791, upper bound: 560.5605809
time: 0.77 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896248, upper bound: 560.5871916
time: 1.32 seconds

## Relational analysis of IS_B2_A1_A2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5896248, upper bound: 560.5881740
time: 1.13 seconds

## BFS IS instance: IS_B2_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -130.8984680, 399.8167725, -188.3446960, 579.6983643, -710.5968018, 588.1614990
1: -184.7044220, 404.4985352, -267.6078186, 586.9411011, -771.6455078, 672.1062622
2: -156.0574341, 446.3069458, -226.3396759, 648.8760376, -804.9334106, 672.6466064
3: -166.6466522, 560.2682495, -241.1611633, 813.9174194, -980.5640259, 801.4294434
4: -139.3530579, 515.3523560, -202.6299744, 750.9100342, -890.2630005, 717.9822388

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5849615, upper bound: 560.5601208
time: 1.16 seconds

## Relational analysis of IS_B2_A1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881072, upper bound: 560.5867564
time: 1.26 seconds

## Relational analysis of IS_B2_A1_A2_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881072, upper bound: 560.5877388
time: 0.91 seconds

## BFS IS instance: IS_B2_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -147.4514923, 449.1673584, -188.3769989, 579.7989502, -727.2504272, 637.5443726
1: -209.1342621, 455.8430176, -267.6529541, 587.0421143, -796.1763916, 723.4959717
2: -176.9795227, 503.3908386, -226.3779755, 648.9873657, -825.9669189, 729.7687988
3: -188.3370514, 630.8516235, -241.2017059, 814.0588989, -1002.3958130, 872.0533447
4: -157.9375305, 581.5914307, -202.6641541, 751.0401001, -908.9776001, 784.2556152

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A2_A1

### Relational analysis result of IS_B2_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5857903, upper bound: 560.5606136
time: 0.76 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889360, upper bound: 560.5872243
time: 1.07 seconds

## Relational analysis of IS_B2_A1_A2_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5889360, upper bound: 560.5882067
time: 1.25 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -176.0407715, 542.0167847, -180.8099976, 556.1011963, -732.1419678, 722.8267822
1: -250.2174072, 548.9858398, -256.9301758, 563.2346191, -813.4520264, 805.9159546
2: -211.7223206, 607.0142212, -217.2973633, 622.8032227, -834.5254517, 824.3114624
3: -225.4848785, 761.2257690, -231.5204010, 780.8549194, -1006.3397827, 992.7461548
4: -189.5153198, 702.1432495, -194.5097809, 720.4786377, -909.9939575, 896.6530151

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5543760
time: 1.02 seconds

## Relational analysis of IS_B2_A2_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5568049
time: 0.89 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -176.0407715, 542.0167847, -282.3350525, 872.2695312, -1045.8802490, 824.3518066
1: -250.2174072, 548.9858398, -400.9938660, 882.0807495, -1128.7263184, 949.6586304
2: -211.7223206, 607.0142212, -338.4806824, 975.1448364, -1182.5576172, 945.3633423
3: -225.4848785, 761.2257690, -361.4702759, 1219.9283447, -1442.7293701, 1122.6959229
4: -189.5153198, 702.1432495, -303.4025574, 1124.5421143, -1312.2239990, 1005.2028198

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5543760
time: 1.26 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5568049
time: 1.12 seconds

## BFS IS instance: IS_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -274.8025513, 851.1666260, -180.8099976, 556.1011963, -830.9036255, 1029.7396240
1: -390.8899536, 860.7320557, -256.9301758, 563.2346191, -953.2219849, 1114.7613525
2: -329.9025574, 951.4403076, -217.2973633, 622.8032227, -952.0178833, 1165.0246582
3: -352.3719177, 1190.1116943, -231.5204010, 780.8549194, -1133.2268066, 1419.3599854
4: -295.7261047, 1096.5043945, -194.5097809, 720.4786377, -1015.5238037, 1289.6402588

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5541559
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5567068
time: 1.09 seconds

## BFS IS instance: IS_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -274.8025513, 851.1666260, -282.3350525, 872.2695312, -1143.6553955, 1130.2291260
1: -390.8899536, 860.7320557, -400.9938660, 882.0807495, -1267.7036133, 1257.0207520
2: -329.9025574, 951.4403076, -338.4806824, 975.1448364, -1299.2971191, 1284.6779785
3: -352.3719177, 1190.1116943, -361.4702759, 1219.9283447, -1568.3001709, 1547.8906250
4: -295.7261047, 1096.5043945, -303.4025574, 1124.5421143, -1417.0262451, 1396.9869385

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5541559
time: 0.70 seconds

## Relational analysis of IS_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5567068
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.34 seconds
IS_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5541903, upper bound: 560.5846086
IS_B1_A1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5484864, upper bound: 560.5472007
IS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5528245, upper bound: 560.5846086
IS_B1_A1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5472007, upper bound: 560.5472007
IS_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5898809
IS_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
IS_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5893775, upper bound: 560.5898809
IS_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5889032, upper bound: 560.5903858
IS_B1_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5898809, upper bound: 560.5889032
IS_B1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5903858, upper bound: 560.5889032
IS_B1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5898809, upper bound: 560.5893775
IS_B1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5903858, upper bound: 560.5896644
IS_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5884004, upper bound: 560.5884004
IS_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5884004, upper bound: 560.5888683
IS_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5892292
IS_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5896971
IS_B1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5807427, upper bound: 560.5775078
IS_B1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5816950, upper bound: 560.5836588
IS_B1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5805618, upper bound: 560.5760622
IS_B1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5816950, upper bound: 560.5844686
IS_B1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5869046, upper bound: 560.5891198
IS_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5869046, upper bound: 560.5898574
IS_B1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5871916, upper bound: 560.5896248
IS_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5871916, upper bound: 560.5903623
IS_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5799119, upper bound: 560.5876220
IS_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5799119, upper bound: 560.5881072
IS_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5788402, upper bound: 560.5772482
IS_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5808445, upper bound: 560.5834463
IS_B1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5731201, upper bound: 560.5783716
IS_B1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5731201, upper bound: 560.5812716
IS_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5889360
IS_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5867564, upper bound: 560.5896736
IS_B2_A1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5775078, upper bound: 560.5807427
IS_B2_A1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5836588, upper bound: 560.5816950
IS_B2_A1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5760622, upper bound: 560.5805618
IS_B2_A1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5844686, upper bound: 560.5816950
IS_B2_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5595502, upper bound: 560.5817515
IS_B2_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5777630, upper bound: 560.5842566
IS_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5783716, upper bound: 560.5731201
IS_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5783716, upper bound: 560.5858766
IS_B2_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5891198, upper bound: 560.5869046
IS_B2_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5891198, upper bound: 560.5878870
IS_B2_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5896248, upper bound: 560.5871916
IS_B2_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5896248, upper bound: 560.5881740
IS_B2_A1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5881072, upper bound: 560.5867564
IS_B2_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5881072, upper bound: 560.5877388
IS_B2_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5889360, upper bound: 560.5872243
IS_B2_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5889360, upper bound: 560.5882067
IS_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5543760
IS_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5568049
IS_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5543760
IS_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5559937, upper bound: 560.5568049
IS_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5541559
IS_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5567068
IS_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5541559
IS_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -560.5554148, upper bound: 560.5567068

## BFS IS instance: IS_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -126.4899063, 388.4345703, -133.9572144, 408.2440796, -534.7338867, 522.3917236
1: -178.4534149, 392.6055908, -189.9827118, 413.8730164, -592.3264160, 582.5882568
2: -150.6621704, 432.9809265, -160.5870209, 456.8753662, -607.5375366, 593.5679321
3: -160.9440765, 543.8226929, -170.8863831, 572.4583130, -733.4022827, 714.7089844
4: -134.6369324, 499.7395325, -143.2717285, 527.5396118, -662.1765137, 643.0111694

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5539440, upper bound: 560.5841037
time: 1.61 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5539440, upper bound: 560.5846086
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -140.2063599, 427.9375916, -133.9996338, 408.3798218, -548.5861816, 561.9372559
1: -199.0193481, 433.9468994, -190.0414886, 414.0076599, -613.0269775, 623.9884033
2: -168.2114410, 479.0105591, -160.6363983, 457.0238037, -625.2351074, 639.6469116
3: -179.0332031, 600.2573853, -170.9391327, 572.6468506, -751.6800537, 771.1963501
4: -150.0622406, 553.1212769, -143.3159027, 527.7120361, -677.7742310, 696.4371948

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5526583, upper bound: 560.5841037
time: 1.03 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5526583, upper bound: 560.5844088
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -154.1096191, 469.6587219, -107.9414673, 328.2743835, -482.3840027, 577.6002197
1: -218.3698120, 476.4615479, -151.8074799, 332.8845215, -551.2542725, 628.2689819
2: -184.7702942, 526.1864624, -128.1442413, 367.7981567, -552.5684814, 654.3306885
3: -196.7025299, 659.5767212, -137.1543121, 460.6455383, -657.3480835, 796.7310181
4: -164.9518280, 608.0355225, -114.6111145, 424.2887878, -589.2406006, 722.6465454

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5837290, upper bound: 560.5542430
time: 1.18 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821115, upper bound: 560.5775115
time: 1.12 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830638, upper bound: 560.5836625
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -122.3909149, 370.8732300, -525.0237427, 592.1812134
1: -218.4280548, 476.5923157, -173.5694885, 377.3158875, -595.7438354, 650.1616211
2: -184.8193207, 526.3300171, -146.6701202, 417.0385742, -601.8579102, 673.0000000
3: -196.7547760, 659.7593384, -156.2867889, 521.4085083, -718.1632690, 816.0461426
4: -164.9955444, 608.2030640, -130.9183655, 481.4752197, -646.4707642, 739.1213379

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836160, upper bound: 560.5528245
time: 0.99 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5819306, upper bound: 560.5760660
time: 0.91 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830638, upper bound: 560.5844723
time: 1.08 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -154.1096191, 469.6587219, -122.1466141, 374.9494019, -529.0589600, 591.8052979
1: -218.3698120, 476.4615479, -172.3555145, 379.0198364, -597.3896484, 648.8170776
2: -184.7702942, 526.1864624, -145.4720612, 417.9904480, -602.7607422, 671.6585083
3: -196.7025299, 659.5767212, -155.4417114, 524.9205933, -721.6231079, 815.0184326
4: -164.9518280, 608.0355225, -129.9905243, 482.3223877, -647.2742310, 738.0260620

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5780020, upper bound: 560.5410562
time: 1.46 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885486, upper bound: 560.5894130
time: 0.97 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885486, upper bound: 560.5898809
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -135.6968994, 413.9247437, -568.0752563, 605.4871826
1: -218.4280548, 476.5923157, -192.6562042, 419.8168030, -638.2448730, 669.2484131
2: -184.8193207, 526.3300171, -162.7923889, 463.3983154, -648.2175903, 689.1223145
3: -196.7547760, 659.7593384, -173.3052979, 580.5690308, -777.3237915, 833.0646362
4: -164.9955444, 608.2030640, -145.2133636, 534.9701538, -699.9656982, 753.4163818

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5899180
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888356, upper bound: 560.5903858
time: 1.28 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -107.9414673, 328.2743835, -154.1096191, 469.6587219, -577.6002197, 482.3840027
1: -151.8074799, 332.8845215, -218.3698120, 476.4615479, -628.2689819, 551.2542725
2: -128.1442413, 367.7981567, -184.7702942, 526.1864624, -654.3306885, 552.5684814
3: -137.1543121, 460.6455383, -196.7025299, 659.5767212, -796.7310181, 657.3480835
4: -114.6111145, 424.2887878, -164.9518280, 608.0355225, -722.6465454, 589.2406006

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5542430, upper bound: 560.5837290
time: 1.11 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5775115, upper bound: 560.5821115
time: 1.22 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836625, upper bound: 560.5830638
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -122.3909149, 370.8732300, -154.1504974, 469.7902832, -592.1812134, 525.0237427
1: -173.5694885, 377.3158875, -218.4280548, 476.5923157, -650.1616211, 595.7438354
2: -146.6701202, 417.0385742, -184.8193207, 526.3300171, -673.0000000, 601.8579102
3: -156.2867889, 521.4085083, -196.7547760, 659.7593384, -816.0461426, 718.1632690
4: -130.9183655, 481.4752197, -164.9955444, 608.2030640, -739.1213379, 646.4707642

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5528245, upper bound: 560.5836160
time: 1.04 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5760660, upper bound: 560.5819306
time: 1.27 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_A2_A2

### Relational analysis result of IS_B1_A1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5844723, upper bound: 560.5830638
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -122.1466141, 374.9494019, -154.1096191, 469.6587219, -591.8052979, 529.0589600
1: -172.3555145, 379.0198364, -218.3698120, 476.4615479, -648.8170776, 597.3896484
2: -145.4720612, 417.9904480, -184.7702942, 526.1864624, -671.6585083, 602.7607422
3: -155.4417114, 524.9205933, -196.7025299, 659.5767212, -815.0184326, 721.6231079
4: -129.9905243, 482.3223877, -164.9518280, 608.0355225, -738.0260620, 647.2742310

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5410562, upper bound: 560.5780020
time: 0.83 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894130, upper bound: 560.5885486
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894130, upper bound: 560.5893775
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -135.6968994, 413.9247437, -154.1504974, 469.7902832, -605.4871826, 568.0752563
1: -192.6562042, 419.8168030, -218.4280548, 476.5923157, -669.2484131, 638.2448730
2: -162.7923889, 463.3983154, -184.8193207, 526.3300171, -689.1223145, 648.2176514
3: -173.3052979, 580.5690308, -196.7547760, 659.7593384, -833.0646362, 777.3237915
4: -145.2133636, 534.9701538, -164.9955444, 608.2030640, -753.4163818, 699.9656982

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5899180, upper bound: 560.5888356
time: 1.02 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5899180, upper bound: 560.5896644
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -134.8042450, 412.6112061, -134.8042450, 412.6112061, -547.4153442, 547.4154053
1: -190.1982117, 417.1871033, -190.1982117, 417.1871033, -607.3853149, 607.3853149
2: -160.6754913, 460.2108765, -160.6754913, 460.2108765, -620.8863525, 620.8863525
3: -171.6001434, 578.0089722, -171.6001434, 578.0089722, -749.6091309, 749.6091309
4: -143.5054932, 531.5460205, -143.5054932, 531.5460205, -675.0515137, 675.0515137

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5740992, upper bound: 560.5539711
time: 1.03 seconds

## Relational analysis of IS_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5874999, upper bound: 560.5874999
time: 1.15 seconds

## BFS IS instance: IS_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -151.5404358, 462.2336121, -134.8042450, 412.6112061, -564.1516113, 597.0378418
1: -214.9032593, 468.8790283, -190.1982117, 417.1871033, -632.0902710, 659.0772705
2: -181.8432465, 517.7068481, -160.6754913, 460.2108765, -642.0540161, 678.3823242
3: -193.5249939, 649.0505981, -171.6001434, 578.0089722, -771.5339355, 820.6507568
4: -162.3022156, 598.2625732, -143.5054932, 531.5460205, -693.8482666, 741.7680664

Time for backsubstitution: 1.71 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=618.8850708007812
rel_dist={0: [-560.590385842507, 560.590385842507]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5880940
time: 1.13 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5880940
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -151.0581512, 461.7381592, -608.3136597, 598.4097900
1: -207.9424896, 453.5178833, -214.3221588, 467.8475647, -675.7900391, 667.8400269
2: -175.7731476, 500.5854492, -181.1492157, 516.3231812, -692.0963135, 681.7346802
3: -187.1182709, 627.4591675, -192.8606873, 647.4821167, -834.6004028, 820.3197632
4: -156.8196411, 578.2070312, -161.6295776, 596.5396118, -753.3591919, 739.8366089

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -149.5736237, 457.5722046, -646.1802979, 730.0730591
1: -267.9796143, 587.7565918, -212.2177429, 463.4609375, -731.4405518, 799.9742432
2: -226.6574402, 649.7827759, -179.3684692, 511.4507446, -738.1081543, 829.1511841
3: -241.4969330, 815.0511475, -190.9532623, 641.4743042, -882.9712524, 1006.0043945
4: -202.9143524, 751.9616089, -160.0287476, 590.9093628, -793.8236694, 911.9903564

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -146.5755005, 447.3516541, -593.9271240, 593.9270630
1: -207.9424896, 453.5178833, -207.9424896, 453.5178833, -661.4603882, 661.4603882
2: -175.7731476, 500.5854492, -175.7731476, 500.5854492, -676.3585815, 676.3585815
3: -187.1182709, 627.4591675, -187.1182709, 627.4591675, -814.5774536, 814.5774536
4: -156.8196411, 578.2070312, -156.8196411, 578.2070312, -735.0266724, 735.0266724

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894659, upper bound: 560.5880940
time: 0.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -188.6081696, 580.4994507, -727.0748901, 635.9598389
1: -207.9424896, 453.5178833, -267.9796143, 587.7565918, -795.6990967, 721.4974976
2: -175.7731476, 500.5854492, -226.6574402, 649.7827759, -825.5559082, 727.2428589
3: -187.1182709, 627.4591675, -241.4969330, 815.0511475, -1002.1694336, 868.9561157
4: -156.8196411, 578.2070312, -202.9143524, 751.9616089, -908.7811890, 781.1213989

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894659, upper bound: 560.5880940
time: 1.21 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -146.5755005, 447.3516541, -635.9598389, 727.0748901
1: -267.9796143, 587.7565918, -207.9424896, 453.5178833, -721.4974976, 795.6990967
2: -226.6574402, 649.7827759, -175.7731476, 500.5854492, -727.2428589, 825.5559082
3: -241.4969330, 815.0511475, -187.1182709, 627.4591675, -868.9561157, 1002.1694336
4: -202.9143524, 751.9616089, -156.8196411, 578.2070312, -781.1213989, 908.7811890

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5853309, upper bound: 560.5751142
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -188.6081696, 580.4994507, -769.1076050, 769.1076050
1: -267.9796143, 587.7565918, -267.9796143, 587.7565918, -855.7362061, 855.7362061
2: -226.6574402, 649.7827759, -226.6574402, 649.7827759, -876.4401855, 876.4401855
3: -241.4969330, 815.0511475, -241.4969330, 815.0511475, -1056.5480957, 1056.5480957
4: -202.9143524, 751.9616089, -202.9143524, 751.9616089, -954.8759155, 954.8759155

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 0.91 seconds

## Relational analysis of IS_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869401, upper bound: 560.5871990
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 8.09 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5894659, upper bound: 560.5880940
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5894659, upper bound: 560.5880940
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5853309, upper bound: 560.5751142
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5869401, upper bound: 560.5871990
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.09
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -142.9491425, 436.0071106, -145.5509644, 444.1470947, -587.0962524, 581.5581055
1: -202.7906494, 442.1823425, -206.4813995, 450.3137207, -653.1042480, 648.6637573
2: -171.3983307, 488.1363220, -174.5328217, 497.0658264, -668.4641724, 662.6690063
3: -182.4763184, 611.6849365, -185.8030701, 622.9982300, -805.4745483, 797.4880371
4: -152.9339752, 563.7099609, -155.7177429, 574.1053467, -727.0393066, 719.4275513

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5893717
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5893717
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -145.2429810, 443.3161316, -597.4666138, 615.0332642
1: -218.4280548, 476.5923157, -206.0613861, 449.4056091, -667.8336792, 682.6535034
2: -184.8193207, 526.3300171, -174.1602325, 496.0017090, -680.8210449, 700.4901733
3: -196.7547760, 659.7593384, -185.4185028, 621.7321167, -818.4868774, 845.1777954
4: -164.9955444, 608.2030640, -155.3764496, 572.8951416, -737.8906860, 763.5794678

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5894831
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5894831
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -142.9491425, 436.0071106, -187.6803741, 577.6834717, -720.6326294, 623.6875000
1: -202.7906494, 442.1823425, -266.6741943, 584.9296875, -787.7202759, 708.8565674
2: -171.3983307, 488.1363220, -225.5447845, 646.6713867, -818.0697021, 713.6810303
3: -182.4763184, 611.6849365, -240.3201447, 811.1128540, -993.5891724, 852.0050659
4: -152.9339752, 563.7099609, -201.9232483, 748.3147583, -901.2487183, 765.6331787

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5857052, upper bound: 560.5601469
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5872955
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -187.2062836, 576.2399902, -730.3905029, 656.9964600
1: -218.4280548, 476.5923157, -265.9987793, 583.4129639, -801.8410034, 742.5909424
2: -184.8193207, 526.3300171, -224.9625854, 644.9472656, -829.7666016, 751.2926025
3: -196.7547760, 659.7593384, -239.7062988, 809.0173340, -1005.7720947, 899.4656372
4: -164.9955444, 608.2030640, -201.3962097, 746.3591309, -911.3546753, 809.5992432

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5854357, upper bound: 560.5605218
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5809985, upper bound: 560.5858294
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894659, upper bound: 560.5880940
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -186.9887390, 575.3843384, -128.8285217, 390.4894104, -577.4781494, 704.2128296
1: -265.6780701, 582.6439819, -182.5141296, 397.0623779, -662.7404175, 765.1579590
2: -224.7295380, 644.1589966, -154.2332153, 438.9310913, -663.6605835, 798.3921509
3: -239.4155731, 807.9306030, -164.3982239, 548.8854980, -788.3010864, 972.3287964
4: -201.1903687, 745.4114990, -137.6930389, 506.8053894, -707.9957275, 883.1044922

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5484037, upper bound: 560.5853790
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872955, upper bound: 560.5900352
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858294, upper bound: 560.5809985
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -185.5286560, 571.2122192, -142.7839813, 435.4452515, -620.9738770, 713.9962158
1: -263.5614014, 578.3868408, -202.5985260, 441.5486145, -705.1098633, 780.9853516
2: -222.9189758, 639.4779053, -171.2445984, 487.3898621, -710.3088379, 810.7225342
3: -237.5664520, 802.0142822, -182.3119965, 610.7657471, -848.3320923, 984.3262939
4: -199.5837402, 740.0799561, -152.7651367, 562.8719482, -762.4556885, 892.8450928

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605218, upper bound: 560.5857052
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5879797, upper bound: 560.5900352
time: 1.23 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880940, upper bound: 560.5894659
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -167.2992096, 512.3647461, -185.1782227, 569.5202637, -736.8194580, 697.5429688
1: -236.6211090, 518.4329834, -263.1308594, 576.7733154, -813.3944092, 781.5637817
2: -200.0903778, 572.9671021, -222.5516052, 637.6853027, -837.7756958, 795.5186768
3: -213.5314636, 718.2940063, -237.1380463, 799.6616821, -1013.1931152, 955.4320679
4: -179.1565399, 662.5183716, -199.2498016, 737.8568115, -917.0133057, 861.7681885

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5575220, upper bound: 560.5825867
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528749, upper bound: 560.5552539
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -184.3911133, 568.1159668, -186.9714966, 575.7932129, -760.1842651, 755.0874634
1: -261.9881287, 575.1311646, -265.7456665, 582.9974365, -844.9855347, 840.8767090
2: -221.6778412, 635.8009033, -224.7991943, 644.5119629, -866.1898193, 860.6000977
3: -236.1141052, 797.7171631, -239.4877472, 808.5018311, -1044.6159668, 1037.2047119
4: -198.4534302, 735.7338867, -201.2471161, 745.8514404, -944.3048096, 936.9810181

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.23 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5893717
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5893717
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5894831
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5894831
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5872955
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5809985, upper bound: 560.5858294
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5894659, upper bound: 560.5880940
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5872955, upper bound: 560.5900352
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5858294, upper bound: 560.5809985
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5879797, upper bound: 560.5900352
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5880940, upper bound: 560.5894659
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5575220, upper bound: 560.5825867
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5528749, upper bound: 560.5552539
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -142.9491425, 436.0071106, -142.9491425, 436.0071106, -578.9562378, 578.9562378
1: -202.7906494, 442.1823425, -202.7906494, 442.1823425, -644.9730225, 644.9730225
2: -171.3983307, 488.1363220, -171.3983307, 488.1363220, -659.5346680, 659.5346069
3: -182.4763184, 611.6849365, -182.4763184, 611.6849365, -794.1612549, 794.1612549
4: -152.9339752, 563.7099609, -152.9339752, 563.7099609, -716.6439209, 716.6439209

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782260, upper bound: 560.5849707
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5887035
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5893717
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -142.9491425, 436.0071106, -154.1504974, 469.7902832, -612.7394409, 590.1575928
1: -202.7906494, 442.1823425, -218.4280548, 476.5923157, -679.3828125, 660.6104126
2: -171.3983307, 488.1363220, -184.8193207, 526.3300171, -697.7283325, 672.9555054
3: -182.4763184, 611.6849365, -196.7547760, 659.7593384, -842.2356567, 808.4396973
4: -152.9339752, 563.7099609, -164.9955444, 608.2030640, -761.1369629, 728.7055054

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5887035
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5893717
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -142.9491425, 436.0071106, -590.1575928, 612.7394409
1: -218.4280548, 476.5923157, -202.7906494, 442.1823425, -660.6104126, 679.3828125
2: -184.8193207, 526.3300171, -171.3983307, 488.1363220, -672.9555054, 697.7283325
3: -196.7547760, 659.7593384, -182.4763184, 611.6849365, -808.4396973, 842.2356567
4: -164.9955444, 608.2030640, -152.9339752, 563.7099609, -728.7055054, 761.1369629

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872715, upper bound: 560.5809985
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5894831
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -154.1504974, 469.7902832, -154.1504974, 469.7902832, -623.9407959, 623.9407959
1: -218.4280548, 476.5923157, -218.4280548, 476.5923157, -695.0202637, 695.0202637
2: -184.8193207, 526.3300171, -184.8193207, 526.3300171, -711.1492920, 711.1493530
3: -196.7547760, 659.7593384, -196.7547760, 659.7593384, -856.5140991, 856.5140991
4: -164.9955444, 608.2030640, -164.9955444, 608.2030640, -773.1986084, 773.1986084

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887963, upper bound: 560.5885363
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887963, upper bound: 560.5894831
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -125.0910950, 378.7121887, -186.1556244, 572.8258057, -697.9168701, 564.8677979
1: -177.2164612, 385.2739868, -264.5025635, 580.0845337, -757.3009033, 649.7765503
2: -149.7499847, 425.9759521, -223.7301483, 641.3475342, -791.0974731, 649.7061157
3: -159.6127014, 532.4531860, -238.3563690, 804.3628540, -963.9755249, 770.8095093
4: -133.6832275, 491.7020264, -200.2986450, 742.1163940, -875.7994995, 692.0006714

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5853790, upper bound: 560.5483994
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894127, upper bound: 560.5872955
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5872955
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -184.5726471, 568.3223267, -706.6991577, 606.3108521
1: -196.3459320, 427.8159790, -262.2158508, 575.4835815, -771.8294678, 690.0316772
2: -165.9172516, 472.2707825, -221.7715149, 636.2849121, -802.2021484, 694.0422974
3: -176.6769562, 591.6629028, -236.3542023, 797.9724121, -974.6492920, 828.0169678
4: -148.0329285, 545.2708130, -198.5616913, 736.3360596, -884.3688965, 743.8325195

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5857052, upper bound: 560.5601469
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894127, upper bound: 560.5875468
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -132.6434021, 400.5512085, -185.5205536, 570.9252930, -703.5686035, 586.0716553
1: -187.8910370, 407.5761414, -263.6041260, 578.0964966, -765.9874268, 671.1802368
2: -159.0097656, 450.6650085, -222.9537659, 639.0985107, -798.1082764, 673.6187744
3: -169.1583252, 563.5170898, -237.5402374, 801.6162720, -970.7745972, 801.0573120
4: -141.8572388, 520.5186768, -199.6008759, 739.5469971, -881.4041748, 720.1195679

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5763557, upper bound: 560.5467615
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5781557, upper bound: 560.5852799
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5808432, upper bound: 560.5858294
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725504, upper bound: 560.5779122
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800911, upper bound: 560.5846934
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800911, upper bound: 560.5858294
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -149.9529419, 456.3767700, -184.1918030, 567.2032471, -717.1561890, 640.5686035
1: -212.5085449, 463.2142944, -261.6758728, 574.2857056, -786.7942505, 724.8901367
2: -179.8332214, 511.6408691, -221.3029633, 634.9068604, -814.7401123, 732.9438477
3: -191.4326782, 641.0862427, -235.8624725, 796.3215942, -987.7542725, 876.9486694
4: -160.5236359, 591.1001587, -198.1382294, 734.7819824, -895.3055420, 789.2384033

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5854357, upper bound: 560.5605218
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5751142
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5880940
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -186.1556244, 572.8258057, -125.0910950, 378.7121887, -564.8677979, 697.9168701
1: -264.5025635, 580.0845337, -177.2164612, 385.2739868, -649.7765503, 757.3009033
2: -223.7301483, 641.3475342, -149.7499847, 425.9759521, -649.7061157, 791.0975342
3: -238.3563690, 804.3628540, -159.6127014, 532.4531860, -770.8095093, 963.9755249
4: -200.2986450, 742.1163940, -133.6832275, 491.7020264, -692.0006714, 875.7994995

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5483994, upper bound: 560.5853790
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872955, upper bound: 560.5894127
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872955, upper bound: 560.5900352
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -185.5205536, 570.9252930, -132.6759491, 400.6557617, -586.1763306, 703.6011963
1: -263.6041260, 578.0964966, -187.9339905, 407.6807861, -671.2847900, 766.0305176
2: -222.9537659, 639.0985107, -159.0463715, 450.7799683, -673.7337646, 798.1448975
3: -237.5402374, 801.6162720, -169.1976929, 563.6649170, -801.2051392, 970.8139648
4: -199.6008759, 739.5469971, -141.8902435, 520.6527710, -720.2536011, 881.4371948

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5467615, upper bound: 560.5763557
time: 1.42 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5852799, upper bound: 560.5781557
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858294, upper bound: 560.5808432
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5779122, upper bound: 560.5725504
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5846934, upper bound: 560.5800911
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858294, upper bound: 560.5809985
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -184.5726471, 568.3223267, -138.3768463, 421.7382202, -606.3108521, 706.6990967
1: -262.2158508, 575.4835815, -196.3459320, 427.8159790, -690.0316772, 771.8294678
2: -221.7715149, 636.2849121, -165.9172516, 472.2707825, -694.0422363, 802.2021484
3: -236.3542023, 797.9724121, -176.6769562, 591.6629028, -828.0169678, 974.6492920
4: -198.5616913, 736.3360596, -148.0329285, 545.2708130, -743.8325195, 884.3688965

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601469, upper bound: 560.5857052
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5875468, upper bound: 560.5894127
time: 1.29 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5879797, upper bound: 560.5900352
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -184.1918030, 567.2032471, -149.9529419, 456.3767700, -640.5686035, 717.1561890
1: -261.6758728, 574.2857056, -212.5085449, 463.2142944, -724.8901367, 786.7942505
2: -221.3029633, 634.9068604, -179.8332214, 511.6408691, -732.9437866, 814.7401123
3: -235.8624725, 796.3215942, -191.4326782, 641.0862427, -876.9486694, 987.7542725
4: -198.1382294, 734.7819824, -160.5236359, 591.1001587, -789.2384033, 895.3056030

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605218, upper bound: 560.5854357
time: 1.29 seconds

## Relational analysis of IS_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5866865
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5894659
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -166.8318481, 510.8707886, -177.8025208, 546.3723755, -713.2041626, 688.6733398
1: -235.9555817, 516.9342041, -252.6424713, 553.5156860, -789.4712524, 769.5766602
2: -199.5284576, 571.3212280, -213.6806946, 612.1157837, -811.6442261, 785.0018311
3: -212.9311676, 716.2003784, -227.6736755, 767.2701416, -980.2012329, 943.8739624
4: -178.6540680, 660.6062012, -191.2928314, 708.0325317, -886.6865234, 851.8990479

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528639, upper bound: 560.5552539
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5528639, upper bound: 560.5552539
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -183.8584747, 566.4506836, -178.8903809, 550.5550537, -734.4133911, 745.3410034
1: -261.2358704, 573.4628906, -254.3045044, 557.6137085, -818.8494263, 827.7673950
2: -221.0413055, 633.9655762, -215.1133270, 616.5635986, -837.6049194, 849.0789185
3: -235.4342499, 795.3864136, -229.1580353, 773.0988159, -1008.5329590, 1024.5444336
4: -197.8813324, 733.5916138, -192.5489960, 713.2316895, -911.1129150, 926.1405640

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -180.3191986, 555.0098267, -279.7139587, 864.9661255, -1042.9392090, 834.7237549
1: -256.1581421, 561.9249878, -397.4779663, 874.6846313, -1127.3382568, 958.8883667
2: -216.7727356, 621.2052612, -335.4912415, 966.9331665, -1179.4508057, 956.3873291
3: -230.8624268, 779.3031006, -358.3034668, 1209.6142578, -1437.8236084, 1137.6064453
4: -194.0708923, 718.8326416, -300.7332153, 1114.8325195, -1307.0772705, 1019.0744629

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
time: 1.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5887035
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5893717
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5887035
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5899481, upper bound: 560.5893717
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5872715, upper bound: 560.5809985
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5893717, upper bound: 560.5894831
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5887963, upper bound: 560.5885363
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5887963, upper bound: 560.5894831
IS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5894127, upper bound: 560.5872955
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5872955
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5894127, upper bound: 560.5875468
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5879797
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5800911, upper bound: 560.5846934
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5800911, upper bound: 560.5858294
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5751142
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5880940
IS_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5872955, upper bound: 560.5894127
IS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5872955, upper bound: 560.5900352
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5846934, upper bound: 560.5800911
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5858294, upper bound: 560.5809985
IS_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5875468, upper bound: 560.5894127
IS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5879797, upper bound: 560.5900352
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5866865
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5894659
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5528639, upper bound: 560.5552539
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5528639, upper bound: 560.5552539
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 0, lower bound: -560.5560781, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -125.0910950, 378.7121887, -140.0280457, 426.7821350, -551.8732300, 518.7402344
1: -177.2164612, 385.2739868, -198.5680237, 433.0234070, -610.2397461, 583.8420410
2: -149.7499847, 425.9759521, -167.8138275, 478.1621399, -627.9121094, 593.7897949
3: -159.6127014, 532.4531860, -178.7206879, 599.0136108, -758.6262817, 711.1738281
4: -133.6832275, 491.7020264, -149.7527771, 552.2443237, -685.9274292, 641.4547729

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5712749, upper bound: 560.5375870
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5428712, upper bound: 560.5334273
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -141.1082458, 430.1938782, -568.5706787, 562.8464355
1: -196.3459320, 427.8159790, -200.2019653, 436.3524170, -632.6983643, 628.0178833
2: -165.9172516, 472.2707825, -169.2023468, 481.7090454, -647.6262817, 641.4730835
3: -176.6769562, 591.6629028, -180.1458893, 603.5477295, -780.2246704, 771.8087769
4: -148.0329285, 545.2708130, -150.9706421, 556.2291870, -704.2620850, 696.2414551

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893093, upper bound: 560.5899481
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893093, upper bound: 560.5899481
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -125.0910950, 378.7121887, -150.8326721, 459.3978882, -584.4890137, 529.5448608
1: -177.2164612, 385.2739868, -213.6336060, 466.2422180, -643.4586182, 598.9075928
2: -149.7499847, 425.9759521, -180.7470856, 515.0252686, -664.7751465, 606.7230225
3: -159.6127014, 532.4531860, -192.4975128, 645.4302368, -805.0429077, 724.9506836
4: -133.6832275, 491.7020264, -161.3991852, 595.2073975, -728.8905029, 653.1011353

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894245, upper bound: 560.5887035
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5900508, upper bound: 560.5887035
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -152.3540955, 463.9478760, -602.3245850, 574.0922241
1: -196.3459320, 427.8159790, -215.8919983, 470.8031921, -667.1491089, 643.7078247
2: -165.9172516, 472.2707825, -182.6870422, 519.9755859, -685.8928223, 654.9578247
3: -176.6769562, 591.6629028, -194.4721069, 651.6606445, -828.3375854, 786.1349487
4: -148.0329285, 545.2708130, -163.0770874, 600.8029175, -748.8358154, 708.3478394

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878831, upper bound: 560.5808650
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878831, upper bound: 560.5893717
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -150.8326721, 459.3978882, -125.0910950, 378.7121887, -529.5448608, 584.4890137
1: -213.6336060, 466.2422180, -177.2164612, 385.2739868, -598.9075928, 643.4586182
2: -180.7470856, 515.0252686, -149.7499847, 425.9759521, -606.7230225, 664.7751465
3: -192.4975128, 645.4302368, -159.6127014, 532.4531860, -724.9506836, 805.0429077
4: -161.3991852, 595.2073975, -133.6832275, 491.7020264, -653.1011353, 728.8905029

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887035, upper bound: 560.5894245
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887035, upper bound: 560.5900508
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -152.3540955, 463.9478760, -138.3768463, 421.7382202, -574.0922241, 602.3245850
1: -215.8919983, 470.8031921, -196.3459320, 427.8159790, -643.7078857, 667.1491089
2: -182.6870422, 519.9755859, -165.9172516, 472.2707825, -654.9578247, 685.8928223
3: -194.4721069, 651.6606445, -176.6769562, 591.6629028, -786.1349487, 828.3375854
4: -163.0770874, 600.8029175, -148.0329285, 545.2708130, -708.3478394, 748.8358154

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5808650, upper bound: 560.5878831
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5808650, upper bound: 560.5900508
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -149.5838776, 454.9882812, -134.8042450, 412.6112061, -562.1950684, 589.7924805
1: -211.9485321, 461.8732910, -190.1982117, 417.1871033, -629.1355591, 652.0714722
2: -179.3557129, 510.1785278, -160.6754913, 460.2108765, -639.5665894, 670.8540039
3: -190.9408417, 639.2671509, -171.6001434, 578.0089722, -768.9497681, 810.8673096
4: -160.1298370, 589.4452515, -143.5054932, 531.5460205, -691.6758423, 732.9506836

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878979, upper bound: 560.5878979
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878979, upper bound: 560.5885363
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -153.1134949, 466.8599854, -151.5404358, 462.2336121, -615.3471069, 618.4003906
1: -217.0741425, 473.6121826, -214.9032593, 468.8790283, -685.9531860, 688.5153809
2: -183.6773682, 522.9962158, -181.8432465, 517.7068481, -701.3840942, 704.8394165
3: -195.5168762, 655.6080933, -193.5249939, 649.0505981, -844.5673828, 849.1330566
4: -163.9641266, 604.3492432, -162.3022156, 598.2625732, -762.2266846, 766.6514893

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884532, upper bound: 560.5887963
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884532, upper bound: 560.5894831
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -107.9414673, 328.2743835, -180.4764709, 554.0835571, -662.0250244, 508.7508545
1: -151.8074799, 332.8845215, -256.4145813, 561.4653931, -713.2728882, 589.2990112
2: -128.1442413, 367.7981567, -216.9042053, 620.8506470, -748.9948730, 584.7023926
3: -137.1543121, 460.6455383, -231.0814819, 778.2500610, -915.4043579, 691.7270508
4: -114.6111145, 424.2887878, -194.1999817, 718.2624512, -832.8734741, 618.4887695

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5847783, upper bound: 560.5483994
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5765595, upper bound: 560.5801038
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824896, upper bound: 560.5815482
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -122.3909149, 370.8732300, -183.2179413, 563.6757202, -686.0666504, 554.0911255
1: -173.5694885, 377.3158875, -260.3947449, 570.9394531, -744.5088501, 637.7104492
2: -146.6701202, 417.0385742, -220.2951813, 631.2506714, -777.9207153, 637.3337402
3: -156.2867889, 521.4085083, -234.6600037, 791.6622314, -947.9489746, 756.0684814
4: -130.9183655, 481.4752197, -197.2195435, 730.4014893, -861.3198242, 678.6947632

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5853790, upper bound: 560.5483994
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5747851, upper bound: 560.5771786
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5840546, upper bound: 560.5815482
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -122.1466141, 374.9494019, -180.7672882, 556.0155640, -678.1621704, 555.7166138
1: -172.3555145, 379.0198364, -256.8242798, 563.1962891, -735.5518188, 635.8441162
2: -145.4720612, 417.9904480, -217.2093353, 622.7513428, -768.2233276, 635.1997681
3: -155.4417114, 524.9205933, -231.5040436, 780.7495728, -936.1912231, 756.4245605
4: -129.9905243, 482.3223877, -194.4868164, 720.5640259, -850.5545044, 676.8092041

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5849985, upper bound: 560.5597875
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886055, upper bound: 560.5864298
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886055, upper bound: 560.5875468
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -135.6968994, 413.9247437, -182.8995514, 563.5199585, -699.2168579, 596.8242798
1: -192.6562042, 419.8168030, -259.9295349, 570.6221924, -763.2783813, 679.7462769
2: -162.7923889, 463.3983154, -219.8695221, 630.8998413, -793.6921997, 683.2678223
3: -173.3052979, 580.5690308, -234.2979279, 791.2804565, -964.5857544, 814.8669434
4: -145.2133636, 534.9701538, -196.8548126, 730.0856934, -875.2990112, 731.8249512

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5857052, upper bound: 560.5601469
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891809, upper bound: 560.5868414
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891809, upper bound: 560.5879797
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -128.0443878, 385.5637817, -164.2853851, 503.0168762, -631.0612793, 549.8491821
1: -181.3908539, 392.7050476, -232.3374329, 509.0014954, -690.3923340, 625.0424805
2: -153.5391998, 434.3264465, -196.4610596, 562.5321045, -716.0711670, 630.7874756
3: -163.3291321, 542.7523804, -209.6621094, 705.1931763, -868.5223389, 752.4144897
4: -136.9945984, 501.5542297, -175.9047546, 650.4016724, -787.3962402, 677.4589844

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5771122, upper bound: 560.5840147
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5800911, upper bound: 560.5846934
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718905, upper bound: 560.5760658
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794137, upper bound: 560.5831097
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793409, upper bound: 560.5826391
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -131.6932068, 398.0427856, -181.3702545, 558.7680054, -690.4611816, 579.4130249
1: -186.7372742, 405.0321045, -257.7017822, 565.6839600, -752.4212646, 662.7338867
2: -158.0387115, 447.8122253, -218.0462189, 625.3452759, -783.3839111, 665.8584595
3: -168.1080170, 559.9805298, -232.2370911, 784.5942993, -952.7023315, 792.2176514
4: -140.9857178, 517.2449951, -195.2057037, 723.5949707, -864.5806885, 712.4506226

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5763557, upper bound: 560.5467615
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5781557, upper bound: 560.5852785
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782062, upper bound: 560.5729199
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5782062, upper bound: 560.5858294
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -149.9529419, 456.3767700, -170.8580017, 522.9378052, -672.8907471, 627.2347412
1: -212.5085449, 463.2142944, -242.9760132, 530.4552002, -742.9637451, 706.1902466
2: -179.8332214, 511.6408691, -205.5863800, 586.7113037, -766.5445557, 717.2272339
3: -191.4326782, 641.0862427, -218.7968597, 734.8660889, -926.2987061, 859.8830566
4: -160.5236359, 591.1001587, -183.9874725, 678.5155029, -839.0391235, 775.0876465

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5613255, upper bound: 560.4995807
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856942, upper bound: 560.5744053
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5751142
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -149.9529419, 456.3767700, -178.8815460, 550.5902710, -700.5432129, 635.2583008
1: -212.5085449, 463.2142944, -254.0803070, 557.5898438, -770.0983887, 717.2945557
2: -179.8332214, 511.6408691, -214.9343109, 616.4430542, -796.2762451, 726.5751343
3: -191.4326782, 641.0862427, -229.0572205, 773.1149292, -964.5476074, 870.1434326
4: -160.5236359, 591.1001587, -192.4313812, 713.5560913, -874.0796509, 783.5315552

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5613255, upper bound: 560.5534181
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5856942, upper bound: 560.5873067
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5877693
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -180.4764709, 554.0835571, -107.9414673, 328.2743835, -508.7508545, 662.0250244
1: -256.4145813, 561.4653931, -151.8074799, 332.8845215, -589.2989502, 713.2728882
2: -216.9042053, 620.8506470, -128.1442413, 367.7981567, -584.7023926, 748.9948730
3: -231.0814819, 778.2500610, -137.1543121, 460.6455383, -691.7270508, 915.4043579
4: -194.1999817, 718.2624512, -114.6111145, 424.2887878, -618.4887695, 832.8734741

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5483994, upper bound: 560.5847783
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801038, upper bound: 560.5765595
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815482, upper bound: 560.5824896
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -183.2179413, 563.6757202, -122.3909149, 370.8732300, -554.0911255, 686.0666504
1: -260.3947449, 570.9394531, -173.5694885, 377.3158875, -637.7104492, 744.5088501
2: -220.2951813, 631.2506714, -146.6701202, 417.0385742, -637.3337402, 777.9207153
3: -234.6600037, 791.6622314, -156.2867889, 521.4085083, -756.0684814, 947.9489746
4: -197.2195435, 730.4014893, -130.9183655, 481.4752197, -678.6947632, 861.3198242

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5483994, upper bound: 560.5853790
time: 1.18 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5771786, upper bound: 560.5747851
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815482, upper bound: 560.5840546
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -164.2853851, 503.0168762, -128.0767670, 385.6688232, -549.9542236, 631.0936279
1: -232.3374329, 509.0014954, -181.4333344, 392.8099060, -625.1473389, 690.4348145
2: -196.4610596, 562.5321045, -153.5755463, 434.4417419, -630.9027710, 716.1074219
3: -209.6621094, 705.1931763, -163.3680115, 542.9007568, -752.5628052, 868.5611572
4: -175.9047546, 650.4016724, -137.0273438, 501.6883850, -677.5930786, 787.4289551

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5840147, upper bound: 560.5771122
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5846934, upper bound: 560.5800911
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5760658, upper bound: 560.5718905
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5831097, upper bound: 560.5794137
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5826391, upper bound: 560.5794029
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -181.3702545, 558.7680054, -131.7249908, 398.1453247, -579.5155640, 690.4929810
1: -257.7017822, 565.6839600, -186.7792053, 405.1345520, -662.8363037, 752.4631348
2: -218.0462189, 625.3452759, -158.0744476, 447.9246826, -665.9708862, 783.4197388
3: -232.2370911, 784.5942993, -168.1464081, 560.1254272, -792.3625488, 952.7407227
4: -195.2057037, 723.5949707, -141.0178833, 517.3760986, -712.5817871, 864.6127319

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5467615, upper bound: 560.5763557
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5852785, upper bound: 560.5781557
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5782062
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5809985
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -180.7672882, 556.0155640, -122.1466141, 374.9494019, -555.7166138, 678.1621704
1: -256.8242798, 563.1962891, -172.3555145, 379.0198364, -635.8441162, 735.5518188
2: -217.2093353, 622.7513428, -145.4720612, 417.9904480, -635.1997681, 768.2233276
3: -231.5040436, 780.7495728, -155.4417114, 524.9205933, -756.4245605, 936.1912231
4: -194.4868164, 720.5640259, -129.9905243, 482.3223877, -676.8092041, 850.5545654

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5597875, upper bound: 560.5849985
time: 1.36 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5861797, upper bound: 560.5886055
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5864298, upper bound: 560.5894127
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -182.8995514, 563.5199585, -135.6968994, 413.9247437, -596.8242798, 699.2168579
1: -259.9295349, 570.6221924, -192.6562042, 419.8168030, -679.7462769, 763.2783813
2: -219.8695221, 630.8998413, -162.7923889, 463.3983154, -683.2678223, 793.6921997
3: -234.2979279, 791.2804565, -173.3052979, 580.5690308, -814.8669434, 964.5857544
4: -196.8548126, 730.0856934, -145.2133636, 534.9701538, -731.8249512, 875.2990112

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601469, upper bound: 560.5857052
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5868414, upper bound: 560.5891809
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5861797, upper bound: 560.5900352
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -170.8580017, 522.9378052, -149.9529419, 456.3767700, -627.2347412, 672.8907471
1: -242.9760132, 530.4552002, -212.5085449, 463.2142944, -706.1902466, 742.9637451
2: -205.5863800, 586.7113037, -179.8332214, 511.6408691, -717.2272339, 766.5445557
3: -218.7968597, 734.8660889, -191.4326782, 641.0862427, -859.8830566, 926.2987061
4: -183.9874725, 678.5155029, -160.5236359, 591.1001587, -775.0876465, 839.0391235

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.4995807, upper bound: 560.5613255
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722859, upper bound: 560.5852144
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5717682, upper bound: 560.5847168
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -178.8815460, 550.5902710, -149.9529419, 456.3767700, -635.2583008, 700.5432129
1: -254.0803070, 557.5898438, -212.5085449, 463.2142944, -717.2946167, 770.0983887
2: -214.9343109, 616.4430542, -179.8332214, 511.6408691, -726.5751953, 796.2762451
3: -229.0572205, 773.1149292, -191.4326782, 641.0862427, -870.1434326, 964.5476074
4: -192.4313812, 713.5560913, -160.5236359, 591.1001587, -783.5315552, 874.0797119

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.4995807, upper bound: 560.5854357
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722859, upper bound: 560.5879795
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5717682, upper bound: 560.5874391
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -176.0407715, 542.0167847, -178.8903809, 550.5550537, -726.5958252, 720.9071655
1: -250.2174072, 548.9858398, -254.3045044, 557.6137085, -807.8311157, 803.2902832
2: -211.7223206, 607.0142212, -215.1133270, 616.5635986, -828.2858887, 822.1275635
3: -225.4848785, 761.2257690, -229.1580353, 773.0988159, -998.5836792, 990.3837891
4: -189.5153198, 702.1432495, -192.5489960, 713.2316895, -902.7470093, 894.6921387

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5829020
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5840291
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -274.8025513, 851.1666260, -178.8903809, 550.5550537, -825.3575439, 1027.8129883
1: -390.8899536, 860.7320557, -254.3045044, 557.6137085, -947.6212769, 1111.8288574
2: -329.9025574, 951.4403076, -215.1133270, 616.5635986, -945.7985840, 1162.5529785
3: -352.3719177, 1190.1116943, -229.1580353, 773.0988159, -1125.4707031, 1416.7833252
4: -295.7261047, 1096.5043945, -192.5489960, 713.2316895, -1008.2633667, 1287.4426270

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5829020
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5840291
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -176.0407715, 542.0167847, -279.7139587, 864.9661255, -1038.6314697, 821.7307129
1: -250.2174072, 548.9858398, -397.4779663, 874.6846313, -1121.3420410, 945.8820801
2: -211.7223206, 607.0142212, -335.4912415, 966.9331665, -1174.3468018, 942.1228638
3: -225.4848785, 761.2257690, -358.3034668, 1209.6142578, -1432.3952637, 1119.5291748
4: -189.5153198, 702.1432495, -300.7332153, 1114.8325195, -1302.4923096, 1002.3566284

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5538703
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5564143
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -274.8025513, 851.1666260, -279.7139587, 864.9661255, -1136.4066162, 1127.6141357
1: -390.8899536, 860.7320557, -397.4779663, 874.6846313, -1260.3193359, 1253.2443848
2: -329.9025574, 951.4403076, -335.4912415, 966.9331665, -1291.0864258, 1281.4375000
3: -352.3719177, 1190.1116943, -358.3034668, 1209.6142578, -1557.9659424, 1544.5550537
4: -295.7261047, 1096.5043945, -300.7332153, 1114.8325195, -1407.2945557, 1394.1408691

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5538703
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5564143
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5712749, upper bound: 560.5375870
IS_A1_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5428712, upper bound: 560.5334273
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5893093, upper bound: 560.5899481
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5893093, upper bound: 560.5899481
IS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5894245, upper bound: 560.5887035
IS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5900508, upper bound: 560.5887035
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5878831, upper bound: 560.5808650
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5878831, upper bound: 560.5893717
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5887035, upper bound: 560.5894245
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5887035, upper bound: 560.5900508
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5808650, upper bound: 560.5878831
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5808650, upper bound: 560.5900508
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5878979, upper bound: 560.5878979
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5878979, upper bound: 560.5885363
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5884532, upper bound: 560.5887963
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5884532, upper bound: 560.5894831
IS_A1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5765595, upper bound: 560.5801038
IS_A1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5824896, upper bound: 560.5815482
IS_A1_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5747851, upper bound: 560.5771786
IS_A1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5840546, upper bound: 560.5815482
IS_A1_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5886055, upper bound: 560.5864298
IS_A1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5886055, upper bound: 560.5875468
IS_A1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5891809, upper bound: 560.5868414
IS_A1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5891809, upper bound: 560.5879797
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5794137, upper bound: 560.5831097
IS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5793409, upper bound: 560.5826391
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5782062, upper bound: 560.5729199
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5782062, upper bound: 560.5858294
IS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5856942, upper bound: 560.5744053
IS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5751142
IS_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5856942, upper bound: 560.5873067
IS_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5866865, upper bound: 560.5877693
IS_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5801038, upper bound: 560.5765595
IS_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5815482, upper bound: 560.5824896
IS_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5771786, upper bound: 560.5747851
IS_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5815482, upper bound: 560.5840546
IS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5831097, upper bound: 560.5794137
IS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5826391, upper bound: 560.5794029
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5782062
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5729199, upper bound: 560.5809985
IS_A2_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5861797, upper bound: 560.5886055
IS_A2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5864298, upper bound: 560.5894127
IS_A2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5868414, upper bound: 560.5891809
IS_A2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5861797, upper bound: 560.5900352
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5722859, upper bound: 560.5852144
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5717682, upper bound: 560.5847168
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5722859, upper bound: 560.5879795
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5717682, upper bound: 560.5874391
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5829020
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5840291
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5829020
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5595271, upper bound: 560.5840291
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5538703
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5564143
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5538703
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -560.5553819, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -116.8777847, 353.4006042, -138.7984161, 422.9511719, -539.8289795, 492.1990356
1: -165.4794769, 359.4257812, -196.8127136, 429.1227112, -594.6021729, 556.2385254
2: -139.8109741, 397.3395691, -166.3328400, 473.8507690, -613.6616821, 563.6724243
3: -149.0167084, 496.6348572, -177.1347504, 593.6004639, -742.6171875, 673.7695923
4: -124.8402710, 458.6513977, -148.4311829, 547.2642212, -672.1044922, 607.0825806

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5428712, upper bound: 560.5334273
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5428712, upper bound: 560.5334273
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -125.0910950, 378.7121887, -517.0890503, 546.8293457
1: -196.3459320, 427.8159790, -177.2164612, 385.2739868, -581.6199341, 605.0322876
2: -165.9172516, 472.2707825, -149.7499847, 425.9759521, -591.8931885, 622.0206909
3: -176.6769562, 591.6629028, -159.6127014, 532.4531860, -709.1301270, 751.2755737
4: -148.0329285, 545.2708130, -133.6832275, 491.7020264, -639.7348633, 678.9539185

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5375870, upper bound: 560.5712749
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5300152, upper bound: 560.5300152
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -138.3768463, 421.7382202, -560.1149902, 560.1149292
1: -196.3459320, 427.8159790, -196.3459320, 427.8159790, -624.1618042, 624.1618652
2: -165.9172516, 472.2707825, -165.9172516, 472.2707825, -638.1879272, 638.1879272
3: -176.6769562, 591.6629028, -176.6769562, 591.6629028, -768.3398438, 768.3397827
4: -148.0329285, 545.2708130, -148.0329285, 545.2708130, -693.3037109, 693.3037109

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5712749, upper bound: 560.5376131
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5300152, upper bound: 560.5300152
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -107.9414673, 328.2743835, -146.2552338, 444.5446777, -552.4860840, 474.5295410
1: -151.8074799, 332.8845215, -207.1485596, 451.4632263, -603.2706909, 540.0330200
2: -128.1442413, 367.7981567, -175.2739258, 498.8217163, -626.9659424, 543.0720825
3: -137.1543121, 460.6455383, -186.6784210, 624.8693848, -762.0236816, 647.3239746
4: -114.6111145, 424.2887878, -156.5322876, 576.4097290, -691.0207520, 580.8210449

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888078, upper bound: 560.5878017
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5888078, upper bound: 560.5887035
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -122.3909149, 370.8732300, -149.7982025, 456.4645996, -578.8555298, 520.6714478
1: -173.5694885, 377.3158875, -212.2861023, 463.2638855, -636.8333740, 589.6018066
2: -146.6701202, 417.0385742, -179.6126709, 511.6923828, -658.3623657, 596.6512451
3: -156.2867889, 521.4085083, -191.2666931, 641.2798462, -797.5665894, 712.6751709
4: -130.9183655, 481.4752197, -160.3746033, 591.3582153, -722.2765503, 641.8498535

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893834, upper bound: 560.5878017
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893834, upper bound: 560.5887035
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -132.6434021, 400.5512085, -538.9280396, 554.3814697
1: -196.3459320, 427.8159790, -187.8910370, 407.5761414, -603.9220581, 615.7068481
2: -165.9172516, 472.2707825, -159.0097656, 450.6650085, -616.5822144, 631.2805176
3: -176.6769562, 591.6629028, -169.1583252, 563.5170898, -740.1940308, 760.8212280
4: -148.0329285, 545.2708130, -141.8572388, 520.5186768, -668.5516357, 687.1279907

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872550, upper bound: 560.5804246
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878831, upper bound: 560.5808650
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -138.3768463, 421.7382202, -149.9529419, 456.3767700, -594.7536011, 571.6911621
1: -196.3459320, 427.8159790, -212.5085449, 463.2142944, -659.5602417, 640.3245239
2: -165.9172516, 472.2707825, -179.8332214, 511.6408691, -677.5581055, 652.1039429
3: -176.6769562, 591.6629028, -191.4326782, 641.0862427, -817.7631836, 783.0955200
4: -148.0329285, 545.2708130, -160.5236359, 591.1001587, -739.1330566, 705.7944336

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872550, upper bound: 560.5888727
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878831, upper bound: 560.5892650
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -146.2552338, 444.5446777, -107.9414673, 328.2743835, -474.5295410, 552.4860840
1: -207.1485596, 451.4632263, -151.8074799, 332.8845215, -540.0330200, 603.2706909
2: -175.2739258, 498.8217163, -128.1442413, 367.7981567, -543.0720825, 626.9659424
3: -186.6784210, 624.8693848, -137.1543121, 460.6455383, -647.3239746, 762.0236816
4: -156.5322876, 576.4097290, -114.6111145, 424.2887878, -580.8210449, 691.0207520

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878017, upper bound: 560.5888078
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878017, upper bound: 560.5894245
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -149.7982025, 456.4645996, -122.3909149, 370.8732300, -520.6714478, 578.8555298
1: -212.2861023, 463.2638855, -173.5694885, 377.3158875, -589.6018066, 636.8333130
2: -179.6126709, 511.6923828, -146.6701202, 417.0385742, -596.6512451, 658.3623657
3: -191.2666931, 641.2798462, -156.2867889, 521.4085083, -712.6751709, 797.5665894
4: -160.3746033, 591.3582153, -130.9183655, 481.4752197, -641.8498535, 722.2765503

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878017, upper bound: 560.5893834
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5878017, upper bound: 560.5900508
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -132.6434021, 400.5512085, -138.3768463, 421.7382202, -554.3814697, 538.9280396
1: -187.8910370, 407.5761414, -196.3459320, 427.8159790, -615.7068481, 603.9220581
2: -159.0097656, 450.6650085, -165.9172516, 472.2707825, -631.2805176, 616.5822144
3: -169.1583252, 563.5170898, -176.6769562, 591.6629028, -760.8212280, 740.1940308
4: -141.8572388, 520.5186768, -148.0329285, 545.2708130, -687.1279907, 668.5516357

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5802222, upper bound: 560.5872550
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5802229, upper bound: 560.5878831
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -149.9529419, 456.3767700, -138.3768463, 421.7382202, -571.6911621, 594.7536011
1: -212.5085449, 463.2142944, -196.3459320, 427.8159790, -640.3245239, 659.5601807
2: -179.8332214, 511.6408691, -165.9172516, 472.2707825, -652.1039429, 677.5581055
3: -191.4326782, 641.0862427, -176.6769562, 591.6629028, -783.0955200, 817.7631836
4: -160.5236359, 591.1001587, -148.0329285, 545.2708130, -705.7944336, 739.1330566

Time for backsubstitution: 1.74 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=618.8850708007812
rel_dist={0: [-560.5900507657058, 560.5900507657057]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836668, upper bound: 560.5582863
time: 1.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 1.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.38 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.38
Output dim: 0, lower bound: -560.5836668, upper bound: 560.5582863
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.38
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -147.5333099, 450.3576050, -150.0870361, 458.5847168, -606.1180420, 600.4444580
1: -209.3132782, 456.5391235, -212.9395905, 464.7144775, -674.0276489, 669.4786377
2: -176.9316559, 503.9251099, -179.9819489, 512.8895264, -689.8211670, 683.9070435
3: -188.3572540, 631.6350098, -191.6178131, 643.0838623, -831.4411011, 823.2527466
4: -157.8749084, 582.0621338, -160.5927429, 592.5314331, -750.4063110, 742.6549072

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.96 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -207.2558289, 632.3292847, -144.7367249, 441.8908081, -649.1465454, 777.0657959
1: -293.6619568, 641.0078735, -205.1960297, 447.5280457, -741.1900024, 846.2038574
2: -247.8776855, 708.6886597, -173.4975891, 493.8259583, -741.7036133, 882.1862183
3: -264.6026917, 885.5980835, -184.6378021, 619.2495117, -883.8521729, 1070.2358398
4: -221.8534088, 817.2335815, -154.8026886, 570.4835815, -792.3369751, 972.0362549

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 1.03 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 1.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -147.5333099, 450.3576050, -147.5333099, 450.3576050, -597.8909302, 597.8909302
1: -209.3132782, 456.5391235, -209.3132782, 456.5391235, -665.8523560, 665.8523560
2: -176.9316559, 503.9251099, -176.9316559, 503.9251099, -680.8567505, 680.8567505
3: -188.3572540, 631.6350098, -188.3572540, 631.6350098, -819.9921875, 819.9921875
4: -157.8749084, 582.0621338, -157.8749084, 582.0621338, -739.9370117, 739.9370117

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836668, upper bound: 560.5574377
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5833294, upper bound: 560.5582863
time: 0.79 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -147.5333099, 450.3576050, -207.2558289, 632.3292847, -779.8625488, 657.6133423
1: -209.3132782, 456.5391235, -293.6619568, 641.0078735, -850.3210449, 750.2010498
2: -176.9316559, 503.9251099, -247.8776855, 708.6886597, -885.6203003, 751.8027344
3: -188.3572540, 631.6350098, -264.6026917, 885.5980835, -1073.9553223, 896.2376709
4: -157.8749084, 582.0621338, -221.8534088, 817.2335815, -975.1085205, 803.9155273

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836668, upper bound: 560.5574377
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5833294, upper bound: 560.5582863
time: 1.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -207.2558289, 632.3292847, -147.5333099, 450.3576050, -657.6133423, 779.8625488
1: -293.6619568, 641.0078735, -209.3132782, 456.5391235, -750.2010498, 850.3210449
2: -247.8776855, 708.6886597, -176.9316559, 503.9251099, -751.8027344, 885.6203003
3: -264.6026917, 885.5980835, -188.3572540, 631.6350098, -896.2376709, 1073.9553223
4: -221.8534088, 817.2335815, -157.8749084, 582.0621338, -803.9155273, 975.1085205

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5205878, upper bound: 560.5386781
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5471484, upper bound: 560.5471484
time: 0.90 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -207.2558289, 632.3292847, -207.2558289, 632.3292847, -839.5849609, 839.5849609
1: -293.6619568, 641.0078735, -293.6619568, 641.0078735, -934.2750854, 934.2750854
2: -247.8776855, 708.6886597, -247.8776855, 708.6886597, -955.8807983, 955.8807983
3: -264.6026917, 885.5980835, -264.6026917, 885.5980835, -1150.2008057, 1150.2008057
4: -221.8534088, 817.2335815, -221.8534088, 817.2335815, -1039.0870361, 1039.0870361

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5205878, upper bound: 560.5386781
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5471484, upper bound: 560.5471484
time: 0.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.43 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5836668, upper bound: 560.5574377
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5833294, upper bound: 560.5582863
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5836668, upper bound: 560.5574377
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5833294, upper bound: 560.5582863
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5205878, upper bound: 560.5386781
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5471484, upper bound: 560.5471484
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5205878, upper bound: 560.5386781
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 6.43
Output dim: 0, lower bound: -560.5471484, upper bound: 560.5471484

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -142.0272980, 432.8376465, -144.4386444, 440.5140686, -582.5413818, 577.2762451
1: -201.5145264, 439.0659790, -204.9192352, 446.7166138, -648.2310791, 643.9851074
2: -170.3594208, 484.7265015, -173.2311401, 493.1302185, -663.4896240, 657.9575195
3: -181.3276215, 607.2824097, -184.4001160, 617.9360352, -799.2636108, 791.6824951
4: -151.9913788, 559.7343140, -154.5632324, 569.4963989, -721.4877319, 714.2975464

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -141.1664429, 431.0726318, -611.8826294, 697.2676392
1: -256.9301758, 563.2346191, -200.2674561, 436.7703857, -693.7004395, 763.5020142
2: -217.2973633, 622.8032227, -169.2718048, 482.0840759, -699.3813477, 792.0748901
3: -231.5204010, 780.8549194, -180.1711426, 604.2847900, -835.8051758, 961.0260620
4: -194.5097809, 720.4786377, -150.9880981, 556.7943115, -751.3040771, 871.4667358

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -142.0272980, 432.8376465, -202.6133423, 617.5084839, -759.5357056, 635.4509277
1: -201.5145264, 439.0659790, -287.0879822, 626.1929321, -827.7073975, 726.1539307
2: -170.3594208, 484.7265015, -242.3423004, 692.3969116, -862.7563477, 727.0687866
3: -181.3276215, 607.2824097, -258.6640930, 864.9067993, -1046.2342529, 865.9464111
4: -151.9913788, 559.7343140, -216.8946533, 798.2589111, -950.2503052, 776.6289673

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684014, upper bound: 560.5229244
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5758331, upper bound: 560.5494361
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -203.3727570, 620.7866821, -801.5966797, 759.4739380
1: -256.9301758, 563.2346191, -288.1311035, 629.2644653, -885.8902588, 851.3657227
2: -217.2973633, 622.8032227, -243.1780701, 695.7202759, -912.2401733, 865.9812622
3: -231.5204010, 780.8549194, -259.6271667, 869.3163452, -1100.8366699, 1040.4820557
4: -194.5097809, 720.4786377, -217.6612854, 802.1063232, -996.6160889, 938.1398926

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5355697, upper bound: 560.5014477
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5833294, upper bound: 560.5582863
time: 1.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.04 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5684014, upper bound: 560.5229244
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5758331, upper bound: 560.5494361
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5355697, upper bound: 560.5014477
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.04
Output dim: 0, lower bound: -560.5833294, upper bound: 560.5582863

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -142.0272980, 432.8376465, -142.0272980, 432.8376465, -574.8648682, 574.8648682
1: -201.5145264, 439.0659790, -201.5145264, 439.0659790, -640.5803833, 640.5803833
2: -170.3594208, 484.7265015, -170.3594208, 484.7265015, -655.0859375, 655.0859375
3: -181.3276215, 607.2824097, -181.3276215, 607.2824097, -788.6099854, 788.6099854
4: -151.9913788, 559.7343140, -151.9913788, 559.7343140, -711.7256470, 711.7257080

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891242, upper bound: 560.5871441
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886086, upper bound: 560.5872636
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -142.0272980, 432.8376465, -180.8099976, 556.1011963, -698.1284790, 613.6476440
1: -201.5145264, 439.0659790, -256.9301758, 563.2346191, -764.7491455, 695.9959717
2: -170.3594208, 484.7265015, -217.2973633, 622.8032227, -793.1626587, 702.0237427
3: -181.3276215, 607.2824097, -231.5204010, 780.8549194, -962.1825562, 838.8027344
4: -151.9913788, 559.7343140, -194.5097809, 720.4786377, -872.4700317, 754.2440796

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891242, upper bound: 560.5871441
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5886086, upper bound: 560.5872636
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -142.0272980, 432.8376465, -613.6476440, 698.1284790
1: -256.9301758, 563.2346191, -201.5145264, 439.0659790, -695.9959717, 764.7491455
2: -217.2973633, 622.8032227, -170.3594208, 484.7265015, -702.0237427, 793.1626587
3: -231.5204010, 780.8549194, -181.3276215, 607.2824097, -838.8027344, 962.1825562
4: -194.5097809, 720.4786377, -151.9913788, 559.7343140, -754.2440796, 872.4700317

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5826450, upper bound: 560.5740886
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -180.8099976, 556.1011963, -736.9111938, 736.9111938
1: -256.9301758, 563.2346191, -256.9301758, 563.2346191, -820.1647949, 820.1647949
2: -217.2973633, 622.8032227, -217.2973633, 622.8032227, -840.1005249, 840.1004639
3: -231.5204010, 780.8549194, -231.5204010, 780.8549194, -1012.3753052, 1012.3753052
4: -194.5097809, 720.4786377, -194.5097809, 720.4786377, -914.9884033, 914.9884033

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794668, upper bound: 560.5748063
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5802406, upper bound: 560.5802406
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -136.7248230, 417.6588745, -193.9276733, 591.6046143, -728.3293457, 611.5865479
1: -193.7894745, 423.4420166, -274.5396118, 599.8371582, -793.6266479, 697.9813232
2: -163.8832855, 467.4143372, -231.8407898, 663.3652344, -827.0496216, 699.2551270
3: -174.4523621, 586.0238647, -247.3670959, 828.8056641, -1003.2580566, 833.3909912
4: -146.2899780, 539.7686768, -207.5111084, 764.6560059, -910.9458618, 747.2797852

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5650558, upper bound: 560.5212325
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668143, upper bound: 560.5218664
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668138, upper bound: 560.5210277
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -137.2880554, 417.1852722, -193.1357880, 588.1170654, -725.4051514, 610.3210449
1: -194.6916351, 423.5296326, -273.4000549, 596.5221558, -791.2136841, 696.9295044
2: -164.6151886, 467.8024902, -230.8490143, 659.8026123, -824.4177856, 698.6514893
3: -175.1664124, 585.7062988, -246.3671875, 824.1970215, -999.3634033, 832.0734863
4: -146.8641205, 540.0643311, -206.6483459, 760.6847534, -907.5487671, 746.7126465

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5758331, upper bound: 560.5494361
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745737, upper bound: 560.5494340
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684014, upper bound: 560.5468796
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -175.4215240, 539.6197510, -194.4653625, 593.0894775, -768.5108643, 734.0850830
1: -249.2075348, 546.6327515, -275.2465210, 601.6138916, -850.8214111, 821.8792725
2: -210.7879181, 604.5156860, -232.2808990, 665.3811646, -875.7221069, 836.7965088
3: -224.6348267, 757.7390747, -248.1921234, 831.0009766, -1055.6357422, 1005.9312134
4: -188.7006378, 699.3690186, -207.9680328, 767.4126587, -956.1132202, 907.3370361

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5689662, upper bound: 560.5388987
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5757615, upper bound: 560.5503000
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.96 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5891242, upper bound: 560.5871441
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5886086, upper bound: 560.5872636
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5891242, upper bound: 560.5871441
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5886086, upper bound: 560.5872636
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5826450, upper bound: 560.5740886
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5794668, upper bound: 560.5748063
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5802406, upper bound: 560.5802406
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5668143, upper bound: 560.5218664
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5668138, upper bound: 560.5210277
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5745737, upper bound: 560.5494340
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5684014, upper bound: 560.5468796
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5689662, upper bound: 560.5388987
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.96
Output dim: 0, lower bound: -560.5757615, upper bound: 560.5503000

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -138.5859222, 422.0157776, -140.0801544, 426.7172852, -565.3032227, 562.0959473
1: -196.6213684, 428.2755127, -198.7443848, 432.9609680, -629.5822754, 627.0198364
2: -166.2037201, 472.8791504, -168.0052948, 478.0248718, -644.2285767, 640.8843994
3: -176.9192352, 592.2631226, -178.8327026, 598.7863770, -775.7055664, 771.0958252
4: -148.3022919, 545.9376221, -149.9025726, 551.9309692, -700.2331543, 695.8402100

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5884693
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5884693
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -149.2222137, 453.9899597, -139.6963501, 425.6937866, -574.9158936, 593.6862793
1: -211.4302673, 460.9051819, -198.2122040, 431.8173218, -643.2475586, 659.1173706
2: -178.9349365, 509.1267700, -167.5416412, 476.6658325, -655.6007080, 676.6682129
3: -190.4692688, 637.8526611, -178.3486938, 597.1651611, -787.6343994, 816.2012329
4: -159.7602539, 588.1481323, -149.4702911, 550.3709717, -710.1312256, 737.6183472

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5886086
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5886086
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -138.5859222, 422.0157776, -179.0686951, 550.7568359, -689.3427734, 601.0844727
1: -196.6213684, 428.2755127, -254.4867249, 557.8758545, -754.4971313, 682.7622070
2: -166.2037201, 472.8791504, -215.2177887, 616.9061890, -783.1098022, 688.0969238
3: -176.9192352, 592.2631226, -229.3147125, 773.3914795, -950.3107300, 821.5778198
4: -148.3022919, 545.9376221, -192.6529236, 713.5872192, -861.8894043, 738.5905762

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891168, upper bound: 560.5842603
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891242, upper bound: 560.5871441
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -149.2222137, 453.9899597, -178.3960724, 548.8117065, -698.0338745, 632.3860474
1: -211.4302673, 460.9051819, -253.5264130, 555.7943726, -767.2246094, 714.4315186
2: -178.9349365, 509.1267700, -214.3856964, 614.5192871, -793.4541016, 723.5122681
3: -190.4692688, 637.8526611, -228.4468842, 770.5192261, -960.9884644, 866.2994385
4: -159.7602539, 588.1481323, -191.9034271, 710.8732300, -870.6334839, 780.0515747

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5803499, upper bound: 560.5749045
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5811680, upper bound: 560.5803072
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -178.4594421, 548.6314087, -122.4775085, 370.1733398, -548.6328125, 671.1088867
1: -253.6013794, 555.8121338, -173.5214691, 376.7825012, -630.3839111, 729.3334961
2: -214.5245056, 614.6573486, -146.6408081, 416.6890869, -631.2134399, 761.2981567
3: -228.5030975, 770.5072632, -156.2866058, 520.6162720, -749.1193848, 926.7938232
4: -192.0195312, 710.9749146, -130.9127808, 480.8741150, -672.8936768, 841.8876953

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5842603, upper bound: 560.5891168
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5831973, upper bound: 560.5801247
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -175.4215240, 539.6197510, -137.7603149, 419.3455811, -594.7669678, 677.3800659
1: -249.2075348, 546.6327515, -195.4910126, 425.5209045, -674.7284546, 742.1237793
2: -210.7879181, 604.5156860, -165.2534943, 469.8197327, -680.6074219, 769.7691650
3: -224.6348267, 757.7390747, -175.9121246, 588.3991699, -813.0339355, 933.6511841
4: -188.7006378, 699.3690186, -147.4291229, 542.4041748, -731.1047363, 846.7980347

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871441, upper bound: 560.5891242
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5872636, upper bound: 560.5886086
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -176.0941162, 542.1740723, -170.7013702, 525.8059082, -701.9000244, 712.8754272
1: -250.0874176, 548.9689331, -242.3448181, 532.4370117, -782.5243530, 791.3137207
2: -211.5645752, 607.0720215, -205.0542297, 588.8994141, -800.4639282, 812.1262207
3: -225.3644562, 761.2755127, -218.4029694, 738.5319824, -963.8964233, 979.6784668
4: -189.3984222, 702.2048340, -183.6241760, 681.2262573, -870.6246948, 885.8288574

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794668, upper bound: 560.5747546
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794113, upper bound: 560.5738685
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793056, upper bound: 560.5728616
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -175.1049500, 538.2405396, -171.2507324, 526.1196899, -701.2246094, 709.4912720
1: -248.7192383, 545.2794189, -243.1491394, 533.1003418, -781.8195801, 788.4285889
2: -210.3839417, 603.0628052, -205.6950989, 589.6735229, -800.0573120, 808.7579346
3: -224.1294556, 756.0855713, -219.1181793, 739.2873535, -963.4166870, 975.2037354
4: -188.3369598, 697.6386108, -184.1490326, 682.1306152, -870.4675903, 881.7875366

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5802406, upper bound: 560.5801698
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801698, upper bound: 560.5801698
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -132.2486115, 402.7738953, -192.3828125, 586.6591187, -718.9077148, 595.1567383
1: -187.3268280, 408.6529846, -272.2846069, 594.8565063, -782.1833496, 680.9373779
2: -158.4716339, 451.0903320, -229.9390564, 657.8704834, -816.1427002, 681.0294189
3: -168.6320801, 565.1480103, -245.3368835, 821.8361816, -990.4682617, 810.4848633
4: -141.4690552, 520.5455933, -205.8142853, 758.2019653, -899.6710205, 726.3596802

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5642028, upper bound: 560.5200078
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668143, upper bound: 560.5218664
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668143, upper bound: 560.5218664
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -133.0188599, 405.5900574, -191.9536438, 585.2506714, -718.2695312, 597.5437012
1: -188.4357605, 411.4251709, -271.6889954, 593.4954224, -781.9311523, 683.1141357
2: -159.3577728, 454.2478638, -229.4415283, 656.3964233, -815.5515137, 683.6893921
3: -169.6371155, 569.2200928, -244.7963409, 819.9205322, -989.5576172, 814.0164185
4: -142.2549896, 524.3763428, -205.3712006, 756.4586182, -898.7135620, 729.7475586

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5641539, upper bound: 560.5189531
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668138, upper bound: 560.5210277
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5668138, upper bound: 560.5210277
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -118.7583694, 362.8546753, -188.8660889, 574.3666382, -693.1249390, 551.7207642
1: -167.3955078, 367.0710144, -267.3156128, 582.7283325, -750.1237183, 634.3865967
2: -141.3776550, 405.0372009, -225.7299957, 644.6385498, -786.0162354, 630.7669678
3: -150.9654694, 508.1733704, -240.8965759, 805.0629272, -956.0283203, 749.0699463
4: -126.3093796, 467.2720032, -202.1046295, 743.2185669, -869.5279541, 669.3766479

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5743541, upper bound: 560.5461353
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5743613, upper bound: 560.5494340
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -134.2670593, 408.1546631, -189.1836395, 576.5635986, -710.8305664, 597.3383179
1: -190.5337524, 414.3618469, -267.9728088, 584.9359741, -775.4697266, 682.3344727
2: -161.1126862, 457.5285339, -226.3062286, 647.0079956, -807.7758179, 683.8346558
3: -171.3856812, 572.8991089, -241.4815521, 808.1311035, -979.5167847, 814.3806763
4: -143.7080994, 528.2194214, -202.5682678, 745.6499023, -889.3578491, 730.7875977

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5755163, upper bound: 560.5468796
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5750186, upper bound: 560.5462281
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5679445, upper bound: 560.5242636
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5751154, upper bound: 560.5456187
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -166.2850800, 512.3926392, -189.3726196, 577.5676270, -743.8526001, 701.7652588
1: -235.9469604, 518.9075317, -267.8542480, 585.8554688, -821.5001831, 786.7617798
2: -199.6347809, 574.0769043, -226.0696106, 648.0618896, -846.9106445, 800.1464844
3: -212.7238312, 719.8029175, -241.5442200, 809.3761597, -1022.0999756, 961.3471069
4: -178.8207092, 664.2092896, -202.4288330, 747.3523560, -926.1109619, 866.6381226

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5670239, upper bound: 560.5270511
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683094, upper bound: 560.5231098
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683094, upper bound: 560.5388987
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -166.1569977, 510.2202759, -189.4423065, 577.3616333, -743.5185547, 699.6625977
1: -235.8475800, 517.1128540, -268.0110168, 585.7603149, -821.6079102, 785.1239014
2: -199.5578308, 572.0314331, -226.1915436, 647.9620361, -847.3258667, 798.2229004
3: -212.5883484, 716.9814453, -241.6812744, 809.2708740, -1021.8592529, 958.6625977
4: -178.6444550, 661.7874756, -202.5302124, 747.3748169, -926.0191040, 864.3175659

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5612521, upper bound: 560.5464838
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5755034, upper bound: 560.5501558
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.06 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5884693
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5884693
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5886086
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5884693, upper bound: 560.5886086
IS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5891168, upper bound: 560.5842603
IS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5891242, upper bound: 560.5871441
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5803499, upper bound: 560.5749045
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5811680, upper bound: 560.5803072
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5842603, upper bound: 560.5891168
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5831973, upper bound: 560.5801247
IS_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5871441, upper bound: 560.5891242
IS_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5872636, upper bound: 560.5886086
IS_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5794113, upper bound: 560.5738685
IS_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5793056, upper bound: 560.5728616
IS_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5802406, upper bound: 560.5801698
IS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5801698, upper bound: 560.5801698
IS_A1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5668143, upper bound: 560.5218664
IS_A1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5668143, upper bound: 560.5218664
IS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5668138, upper bound: 560.5210277
IS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5668138, upper bound: 560.5210277
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5743541, upper bound: 560.5461353
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5743613, upper bound: 560.5494340
IS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5679445, upper bound: 560.5242636
IS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5751154, upper bound: 560.5456187
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5683094, upper bound: 560.5231098
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5683094, upper bound: 560.5388987
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5612521, upper bound: 560.5464838
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 0, lower bound: -560.5755034, upper bound: 560.5501558

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -138.5859222, 422.0157776, -138.5859222, 422.0157776, -560.6016846, 560.6016846
1: -196.6213684, 428.2755127, -196.6213684, 428.2755127, -624.8968506, 624.8967896
2: -166.2037201, 472.8791504, -166.2037201, 472.8791504, -639.0827026, 639.0827637
3: -176.9192352, 592.2631226, -176.9192352, 592.2631226, -769.1823730, 769.1823730
4: -148.3022919, 545.9376221, -148.3022919, 545.9376221, -694.2398682, 694.2398682

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5806308, upper bound: 560.5793991
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812637, upper bound: 560.5812637
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -138.5859222, 422.0157776, -149.2222137, 453.9899597, -592.5758667, 571.2379761
1: -196.6213684, 428.2755127, -211.4302673, 460.9051819, -657.5263672, 639.7057495
2: -166.2037201, 472.8791504, -178.9349365, 509.1267700, -675.3302612, 651.8139648
3: -176.9192352, 592.2631226, -190.4692688, 637.8526611, -814.7719116, 782.7323608
4: -148.3022919, 545.9376221, -159.7602539, 588.1481323, -736.4503174, 705.6978760

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5806308, upper bound: 560.5793991
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812637, upper bound: 560.5812637
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -149.2222137, 453.9899597, -138.5859222, 422.0157776, -571.2379761, 592.5758667
1: -211.4302673, 460.9051819, -196.6213684, 428.2755127, -639.7057495, 657.5263672
2: -178.9349365, 509.1267700, -166.2037201, 472.8791504, -651.8139648, 675.3302612
3: -190.4692688, 637.8526611, -176.9192352, 592.2631226, -782.7323608, 814.7719116
4: -159.7602539, 588.1481323, -148.3022919, 545.9376221, -705.6978760, 736.4503174

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793991, upper bound: 560.5806308
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812637, upper bound: 560.5812684
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -149.2222137, 453.9899597, -149.2222137, 453.9899597, -603.2121582, 603.2121582
1: -211.4302673, 460.9051819, -211.4302673, 460.9051819, -672.3353882, 672.3353882
2: -178.9349365, 509.1267700, -178.9349365, 509.1267700, -688.0614624, 688.0614624
3: -190.4692688, 637.8526611, -190.4692688, 637.8526611, -828.3218994, 828.3218994
4: -159.7602539, 588.1481323, -159.7602539, 588.1481323, -747.9083862, 747.9083862

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793991, upper bound: 560.5806308
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5812637, upper bound: 560.5812684
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -118.9165497, 359.0091858, -176.9598694, 543.9460449, -662.8626099, 535.9690552
1: -168.4920807, 365.5959778, -251.4818115, 551.1491699, -719.6411133, 617.0776978
2: -142.3878174, 404.3906860, -212.7280731, 609.5382080, -751.9259644, 617.1185913
3: -151.7389526, 505.0253601, -226.5983582, 764.0123901, -915.7512817, 731.6237183
4: -127.1054535, 466.5332031, -190.4237213, 705.0017090, -832.1071167, 656.9568481

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5884427, upper bound: 560.5841924
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891168, upper bound: 560.5842603
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -133.4538574, 405.8472595, -173.5072784, 533.7783203, -667.2321777, 579.3545532
1: -189.3868713, 412.0382690, -246.5028534, 540.7641602, -730.1510010, 658.5410156
2: -160.0541382, 454.9888306, -208.4782562, 598.0592041, -758.1132202, 663.4671021
3: -170.4111786, 569.6350098, -222.1947327, 749.5777588, -919.9889526, 791.8297119
4: -142.8141632, 525.1412964, -186.6430359, 691.8209839, -834.6351318, 711.7843018

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5801012, upper bound: 560.5749016
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5811680, upper bound: 560.5803029
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -144.0258789, 438.8929443, -168.7081909, 519.7744141, -663.8001709, 607.6010132
1: -203.9723816, 445.4166870, -239.5248718, 526.2909546, -730.2633057, 684.9415283
2: -172.6583862, 491.9520569, -202.6414032, 582.0775146, -754.7359009, 694.5933838
3: -183.7772827, 616.6820068, -215.8568115, 730.0023193, -913.7796021, 832.5388184
4: -154.1790466, 568.4356079, -181.4668274, 673.3093262, -827.4883423, 749.9024658

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5785851, upper bound: 560.5744829
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5785851, upper bound: 560.5749045
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.1762695, 437.7011414, -168.9678802, 519.2570190, -663.4331055, 606.6690063
1: -204.0941010, 444.6321716, -239.9281769, 526.0971680, -730.1911621, 684.5602417
2: -172.7500000, 491.3259888, -202.9433136, 581.8656616, -754.6156616, 694.2692871
3: -183.8669586, 615.3269653, -216.2097015, 729.5659790, -913.4329224, 831.5366821
4: -154.2554016, 567.4541626, -181.6779938, 673.0743408, -827.3297119, 749.1320801

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5806378, upper bound: 560.5803040
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5811175, upper bound: 560.5803072
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -176.9598694, 543.9460449, -118.9165497, 359.0091858, -535.9690552, 662.8626099
1: -251.4818115, 551.1491699, -168.4920807, 365.5959778, -617.0776978, 719.6411133
2: -212.7280731, 609.5382080, -142.3878174, 404.3906860, -617.1185913, 751.9259644
3: -226.5983582, 764.0123901, -151.7389526, 505.0253601, -731.6237183, 915.7512817
4: -190.4237213, 705.0017090, -127.1054535, 466.5332031, -656.9569092, 832.1071167

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5841924, upper bound: 560.5884427
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5842603, upper bound: 560.5891168
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -175.6843109, 540.3205566, -126.9268188, 382.1243591, -557.8086548, 667.2472534
1: -249.6798859, 547.3352051, -179.7907257, 389.2431641, -638.9229736, 727.1258545
2: -211.1684113, 605.2256470, -152.2046661, 430.5368958, -641.7053223, 757.4301758
3: -224.9626770, 758.7421875, -161.8647461, 537.8960571, -762.8587646, 920.6068115
4: -189.0196075, 700.0075684, -135.7985229, 497.0302734, -686.0498657, 835.8060913

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5748634, upper bound: 560.5715771
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821872, upper bound: 560.5793462
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822722, upper bound: 560.5769627
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825141, upper bound: 560.5793848
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5819332, upper bound: 560.5793816
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -173.5072784, 533.7783203, -133.4538574, 405.8472595, -579.3545532, 667.2321777
1: -246.5028534, 540.7641602, -189.3868713, 412.0382690, -658.5410767, 730.1510010
2: -208.4782562, 598.0592041, -160.0541382, 454.9888306, -663.4671021, 758.1132812
3: -222.1947327, 749.5777588, -170.4111786, 569.6350098, -791.8297119, 919.9889526
4: -186.6430359, 691.8209839, -142.8141632, 525.1412964, -711.7843018, 834.6351318

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5749016, upper bound: 560.5801012
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5803029, upper bound: 560.5811680
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -173.0901184, 532.6697388, -144.8189697, 439.9482422, -613.0383301, 677.4887085
1: -245.9271088, 539.5161133, -205.2349243, 446.8749695, -692.8018188, 744.7510376
2: -207.9780121, 596.5782471, -173.7096558, 493.7109985, -701.6890259, 770.2878418
3: -221.6714783, 747.8549805, -184.8893127, 618.2851562, -839.9566040, 932.7442017
4: -186.1845093, 690.1550293, -155.0635376, 570.2045898, -756.3890381, 845.2185059

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726986, upper bound: 560.5851129
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726986, upper bound: 560.5886086
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -165.9394989, 515.4168091, -165.5583649, 519.9601440, -685.8996582, 680.9751587
1: -235.7372742, 521.1035156, -235.3557434, 524.6118774, -760.3491211, 756.4592285
2: -199.4515686, 576.0753784, -199.0877380, 579.7923584, -779.2438965, 775.1630859
3: -212.5400238, 723.3552246, -212.2680206, 729.3022461, -941.8421631, 935.6232300
4: -178.7704315, 666.5201416, -178.5787354, 670.9520874, -849.7224731, 845.0988159

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5794113, upper bound: 560.5738147
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732503, upper bound: 560.5722461
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5732503, upper bound: 560.5728616
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -174.0587921, 535.7524414, -167.6699982, 516.2610474, -690.3198242, 703.4224243
1: -247.1860352, 542.5472412, -238.0138702, 522.8602295, -770.0460815, 780.5610352
2: -209.1079712, 600.0203247, -201.3836212, 578.3580322, -787.4658203, 801.4039307
3: -222.7602234, 752.3291016, -214.5118256, 725.2090454, -947.9690552, 966.8409424
4: -187.2105103, 693.9726562, -180.3509674, 668.9288330, -856.1393433, 874.3235474

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5757960, upper bound: 560.5718188
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5786753, upper bound: 560.5728616
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -172.3520813, 529.7939453, -182.8792877, 558.6311646, -730.9831543, 712.6732178
1: -244.8177948, 536.7002563, -259.6720276, 566.6857300, -811.5035400, 796.3723145
2: -207.0723724, 593.5197144, -219.6744690, 626.8107300, -833.8831177, 813.1941528
3: -220.6129761, 744.1163330, -233.9758301, 785.1802979, -1005.7932739, 978.0921631
4: -185.3697510, 686.5341187, -196.8172913, 725.2594604, -910.6292114, 883.3513794

Time for backsubstitution: 1.70 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=618.8850708007812
rel_dist={0: [-560.5891721066854, 560.5891721066855]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1118.78 seconds
