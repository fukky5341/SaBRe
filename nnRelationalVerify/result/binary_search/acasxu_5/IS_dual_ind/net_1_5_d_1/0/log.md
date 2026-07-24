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
execution time: IAR + LP analysis = 1.66 + 2.24 = 3.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -560.5903858, upper bound: 560.5903858


# Binary Search by BASE starts (time budget: 1196.10 seconds, max iter: 100)

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
Binary search time: 76.12 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1119.98 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5882067
time: 1.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -560.5903623, upper bound: 560.5882067
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.13
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -152.4975433, 466.3875427, -612.9630127, 599.8491211
1: -207.9424896, 453.5178833, -216.3767242, 472.4903870, -680.4328613, 669.8945923
2: -175.7731476, 500.5854492, -182.8786926, 521.4256592, -697.1987915, 683.4639893
3: -187.1182709, 627.4591675, -194.7117004, 653.9572754, -841.0755005, 822.1708984
4: -156.8196411, 578.2070312, -163.1766510, 602.4576416, -759.2772217, 741.3836670

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -151.8643036, 464.4644775, -653.0726318, 732.3637085
1: -267.9796143, 587.7565918, -215.4748840, 470.5242004, -738.5037842, 803.2314453
2: -226.6574402, 649.7827759, -182.1171265, 519.2548828, -745.9122314, 831.8997803
3: -241.4969330, 815.0511475, -193.8968964, 651.2348633, -892.7318115, 1008.9479370
4: -202.9143524, 751.9616089, -162.4932098, 599.9415283, -802.8558350, 914.4547729

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.12 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 0, lower bound: -560.5881831, upper bound: 560.5881831

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -146.5755005, 447.3516541, -593.9271240, 593.9270630
1: -207.9424896, 453.5178833, -207.9424896, 453.5178833, -661.4603882, 661.4603882
2: -175.7731476, 500.5854492, -175.7731476, 500.5854492, -676.3585815, 676.3585815
3: -187.1182709, 627.4591675, -187.1182709, 627.4591675, -814.5774536, 814.5774536
4: -156.8196411, 578.2070312, -156.8196411, 578.2070312, -735.0266724, 735.0266724

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5843363, upper bound: 560.5871481
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5877376
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -188.6081696, 580.4994507, -727.0748901, 635.9598389
1: -207.9424896, 453.5178833, -267.9796143, 587.7565918, -795.6990967, 721.4974976
2: -175.7731476, 500.5854492, -226.6574402, 649.7827759, -825.5559082, 727.2428589
3: -187.1182709, 627.4591675, -241.4969330, 815.0511475, -1002.1694336, 868.9561157
4: -156.8196411, 578.2070312, -202.9143524, 751.9616089, -908.7811890, 781.1213989

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5843363, upper bound: 560.5871481
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5877376
time: 1.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -146.5755005, 447.3516541, -635.9598389, 727.0748901
1: -267.9796143, 587.7565918, -207.9424896, 453.5178833, -721.4974976, 795.6990967
2: -226.6574402, 649.7827759, -175.7731476, 500.5854492, -727.2428589, 825.5559082
3: -241.4969330, 815.0511475, -187.1182709, 627.4591675, -868.9561157, 1002.1694336
4: -202.9143524, 751.9616089, -156.8196411, 578.2070312, -781.1213989, 908.7811890

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5831046, upper bound: 560.5873032
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5877141, upper bound: 560.5877141
time: 0.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -188.6081696, 580.4994507, -769.1076050, 769.1076050
1: -267.9796143, 587.7565918, -267.9796143, 587.7565918, -855.7362061, 855.7362061
2: -226.6574402, 649.7827759, -226.6574402, 649.7827759, -876.4401855, 876.4401855
3: -241.4969330, 815.0511475, -241.4969330, 815.0511475, -1056.5480957, 1056.5480957
4: -202.9143524, 751.9616089, -202.9143524, 751.9616089, -954.8759155, 954.8759155

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5831046, upper bound: 560.5873032
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5877141, upper bound: 560.5877141
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5843363, upper bound: 560.5871481
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5877376
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5843363, upper bound: 560.5871481
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5877376
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5831046, upper bound: 560.5873032
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5877141, upper bound: 560.5877141
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5831046, upper bound: 560.5873032
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -560.5877141, upper bound: 560.5877141

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -145.4742279, 444.5320435, -577.2070923, 553.3185425
1: -188.4647369, 414.0709229, -206.3853912, 450.5606689, -639.0253906, 620.4562378
2: -159.2985992, 457.8714600, -174.4485168, 497.3120728, -656.6105957, 632.3199463
3: -169.7159882, 574.7191162, -185.7286682, 623.4815674, -793.1975708, 760.4477539
4: -142.6433258, 530.3728638, -155.6707153, 574.4890137, -717.1323242, 686.0435791

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5837704, upper bound: 560.5837704
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5837704, upper bound: 560.5891923
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -146.5755005, 447.3516541, -588.3191528, 576.9470215
1: -199.8453827, 436.4012756, -207.9424896, 453.5178833, -653.3632812, 644.3437500
2: -168.9626923, 481.6351929, -175.7731476, 500.5854492, -669.5480347, 657.4083252
3: -179.8964081, 603.5338135, -187.1182709, 627.4591675, -807.3555298, 790.6520996
4: -150.7288818, 555.7738037, -156.8196411, 578.2070312, -728.9359131, 712.5934448

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891923, upper bound: 560.5843598
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891923, upper bound: 560.5897817
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -187.2031097, 576.8488159, -709.5239258, 595.0474243
1: -188.4647369, 414.0709229, -266.0174561, 583.9544067, -772.4191284, 680.0882568
2: -159.2985992, 457.8714600, -225.0015564, 645.5560913, -804.8545532, 682.8730469
3: -169.7159882, 574.7191162, -239.7432861, 809.8814087, -979.5974121, 814.4624023
4: -142.6433258, 530.3728638, -201.4611053, 747.1083984, -889.7517090, 731.8339233

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5839254, upper bound: 560.5825387
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5839254, upper bound: 560.5871481
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -188.6081696, 580.4994507, -721.4669189, 618.9797363
1: -199.8453827, 436.4012756, -267.9796143, 587.7565918, -787.6019897, 704.3808594
2: -168.9626923, 481.6351929, -226.6574402, 649.7827759, -818.7453613, 708.2926025
3: -179.8964081, 603.5338135, -241.4969330, 815.0511475, -994.9475098, 845.0307617
4: -150.7288818, 555.7738037, -202.9143524, 751.9616089, -902.6904907, 758.6881714

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5831281
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5877376
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -182.5155792, 572.6014404, -145.4742279, 444.5320435, -627.0474854, 718.0756226
1: -259.6409302, 577.7786865, -206.3853912, 450.5606689, -710.2015991, 784.1639404
2: -219.5419617, 638.3512573, -174.4485168, 497.3120728, -716.8539429, 812.7998047
3: -234.1709137, 803.0667114, -185.7286682, 623.4815674, -857.6523438, 988.7954102
4: -196.8828735, 739.0787354, -155.6707153, 574.4890137, -771.3718262, 894.7494507

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5571587, upper bound: 560.5858749
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825387, upper bound: 560.5839254
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825387, upper bound: 560.5893473
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -185.2178802, 569.7943726, -146.5755005, 447.3516541, -632.5695190, 716.3698120
1: -263.1491394, 577.0369263, -207.9424896, 453.5178833, -716.6669922, 784.9794312
2: -222.5697784, 637.9903564, -175.7731476, 500.5854492, -723.1552124, 813.7634888
3: -237.1614532, 800.1052856, -187.1182709, 627.4591675, -864.6206055, 987.2235718
4: -199.2773132, 738.2121582, -156.8196411, 578.2070312, -777.4843750, 895.0317383

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5606136, upper bound: 560.5858749
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871481, upper bound: 560.5843363
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871481, upper bound: 560.5897582
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -182.5155792, 572.6014404, -187.2031097, 576.8488159, -759.3643188, 759.8045044
1: -259.6409302, 577.7786865, -266.0174561, 583.9544067, -843.5953369, 843.7960205
2: -219.5419617, 638.3512573, -225.0015564, 645.5560913, -865.0979614, 863.3527832
3: -234.1709137, 803.0667114, -239.7432861, 809.8814087, -1044.0523682, 1042.8100586
4: -196.8828735, 739.0787354, -201.4611053, 747.1083984, -943.9912720, 940.5397339

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5571352, upper bound: 560.5838308
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -185.2178802, 569.7943726, -188.6081696, 580.4994507, -765.7173462, 758.4025269
1: -263.1491394, 577.0369263, -267.9796143, 587.7565918, -850.9055786, 845.0165405
2: -222.5697784, 637.9903564, -226.6574402, 649.7827759, -872.3525391, 864.6478271
3: -237.1614532, 800.1052856, -241.4969330, 815.0511475, -1052.2126465, 1041.6021729
4: -199.2773132, 738.2121582, -202.9143524, 751.9616089, -951.2388916, 941.1264648

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605900, upper bound: 560.5838308
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5837704, upper bound: 560.5837704
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5837704, upper bound: 560.5891923
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5891923, upper bound: 560.5843598
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5891923, upper bound: 560.5897817
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5839254, upper bound: 560.5825387
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5839254, upper bound: 560.5871481
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5831281
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5877376
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5825387, upper bound: 560.5839254
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5825387, upper bound: 560.5893473
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5871481, upper bound: 560.5843363
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5871481, upper bound: 560.5897582
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5571352, upper bound: 560.5838308
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5605900, upper bound: 560.5838308
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.27
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -132.6752014, 407.8443604, -540.5195312, 540.5194702
1: -188.4647369, 414.0709229, -188.4647369, 414.0709229, -602.5356445, 602.5356445
2: -159.2985992, 457.8714600, -159.2985992, 457.8714600, -617.1700439, 617.1700439
3: -169.7159882, 574.7191162, -169.7159882, 574.7191162, -744.4350586, 744.4350586
4: -142.6433258, 530.3728638, -142.6433258, 530.3728638, -673.0161743, 673.0161743

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5797149, upper bound: 560.5674417
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -140.9674835, 430.3715820, -563.0466919, 548.8118286
1: -188.4647369, 414.0709229, -199.8453827, 436.4012756, -624.8660278, 613.9163208
2: -159.2985992, 457.8714600, -168.9626923, 481.6351929, -640.9337769, 626.8341675
3: -169.7159882, 574.7191162, -179.8964081, 603.5338135, -773.2498169, 754.6154175
4: -142.6433258, 530.3728638, -150.7288818, 555.7738037, -698.4171143, 681.1017456

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5797149, upper bound: 560.5789173
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5748618
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -132.6752014, 407.8443604, -548.8118286, 563.0466919
1: -199.8453827, 436.4012756, -188.4647369, 414.0709229, -613.9163208, 624.8660278
2: -168.9626923, 481.6351929, -159.2985992, 457.8714600, -626.8341675, 640.9337769
3: -179.8964081, 603.5338135, -169.7159882, 574.7191162, -754.6154175, 773.2497559
4: -150.7288818, 555.7738037, -142.6433258, 530.3728638, -681.1017456, 698.4171143

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793880, upper bound: 560.5688676
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5707075, upper bound: 560.5668220
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -140.9674835, 430.3715820, -571.3390503, 571.3390503
1: -199.8453827, 436.4012756, -199.8453827, 436.4012756, -636.2466431, 636.2466431
2: -168.9626923, 481.6351929, -168.9626923, 481.6351929, -650.5977783, 650.5977783
3: -179.8964081, 603.5338135, -179.8964081, 603.5338135, -783.4301758, 783.4301758
4: -150.7288818, 555.7738037, -150.7288818, 555.7738037, -706.5026855, 706.5026855

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793880, upper bound: 560.5728390
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5707075, upper bound: 560.5710801
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -182.5155792, 572.6014404, -705.2765503, 590.3598633
1: -188.4647369, 414.0709229, -259.6409302, 577.7786865, -766.2434082, 673.7118530
2: -159.2985992, 457.8714600, -219.5419617, 638.3512573, -797.6497803, 677.4134521
3: -169.7159882, 574.7191162, -234.1709137, 803.0667114, -972.7826538, 808.8898315
4: -142.6433258, 530.3728638, -196.8828735, 739.0787354, -881.7220459, 727.2557373

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5804530, upper bound: 560.5565693
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5766031, upper bound: 560.5737768
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5738353
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -185.2178802, 569.7943726, -702.4694824, 593.0621948
1: -188.4647369, 414.0709229, -263.1491394, 577.0369263, -765.5016479, 677.2198486
2: -159.2985992, 457.8714600, -222.5697784, 637.9903564, -797.2889404, 680.4412231
3: -169.7159882, 574.7191162, -237.1614532, 800.1052856, -969.8212280, 811.8805542
4: -142.6433258, 530.3728638, -199.2773132, 738.2121582, -880.8554688, 729.6501465

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5804530, upper bound: 560.5600241
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5766031, upper bound: 560.5798900
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5799485
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -182.5155792, 572.6014404, -713.5689087, 612.8870850
1: -199.8453827, 436.4012756, -259.6409302, 577.7786865, -777.6240845, 696.0422363
2: -168.9626923, 481.6351929, -219.5419617, 638.3512573, -807.3138428, 701.1770630
3: -179.8964081, 603.5338135, -234.1709137, 803.0667114, -982.9630737, 837.7045898
4: -150.7288818, 555.7738037, -196.8828735, 739.0787354, -889.8076172, 752.6566772

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858749, upper bound: 560.5571587
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5823342
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5831281
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -185.2178802, 569.7943726, -710.7618408, 615.5894165
1: -199.8453827, 436.4012756, -263.1491394, 577.0369263, -776.8823242, 699.5502930
2: -168.9626923, 481.6351929, -222.5697784, 637.9903564, -806.9529419, 704.2049561
3: -179.8964081, 603.5338135, -237.1614532, 800.1052856, -980.0016479, 840.6952515
4: -150.7288818, 555.7738037, -199.2773132, 738.2121582, -888.9410400, 755.0511475

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858749, upper bound: 560.5606136
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5869437
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5876440
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -182.5155792, 572.6014404, -132.6752014, 407.8443604, -590.3598633, 705.2765503
1: -259.6409302, 577.7786865, -188.4647369, 414.0709229, -673.7118530, 766.2434082
2: -219.5419617, 638.3512573, -159.2985992, 457.8714600, -677.4134521, 797.6497803
3: -234.1709137, 803.0667114, -169.7159882, 574.7191162, -808.8898315, 972.7826538
4: -196.8828735, 739.0787354, -142.6433258, 530.3728638, -727.2557373, 881.7220459

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728483, upper bound: 560.5815241
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824776, upper bound: 560.5839254
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -182.5155792, 572.6014404, -140.9674835, 430.3715820, -612.8871460, 713.5689087
1: -259.6409302, 577.7786865, -199.8453827, 436.4012756, -696.0422363, 777.6240845
2: -219.5419617, 638.3512573, -168.9626923, 481.6351929, -701.1770630, 807.3138428
3: -234.1709137, 803.0667114, -179.8964081, 603.5338135, -837.7045898, 982.9630737
4: -196.8828735, 739.0787354, -150.7288818, 555.7738037, -752.6566772, 889.8075562

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728483, upper bound: 560.5869460
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5824776, upper bound: 560.5893473
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -185.2178802, 569.7943726, -132.6752014, 407.8443604, -593.0621948, 702.4694824
1: -263.1491394, 577.0369263, -188.4647369, 414.0709229, -677.2199097, 765.5016479
2: -222.5697784, 637.9903564, -159.2985992, 457.8714600, -680.4412231, 797.2888794
3: -237.1614532, 800.1052856, -169.7159882, 574.7191162, -811.8805542, 969.8212891
4: -199.2773132, 738.2121582, -142.6433258, 530.3728638, -729.6501465, 880.8554688

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5787930, upper bound: 560.5688522
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5860222, upper bound: 560.5835987
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871431, upper bound: 560.5843363
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -185.2178802, 569.7943726, -140.9674835, 430.3715820, -615.5894775, 710.7618408
1: -263.1491394, 577.0369263, -199.8453827, 436.4012756, -699.5503540, 776.8823242
2: -222.5697784, 637.9903564, -168.9626923, 481.6351929, -704.2049561, 806.9529419
3: -237.1614532, 800.1052856, -179.8964081, 603.5338135, -840.6952515, 980.0016479
4: -199.2773132, 738.2121582, -150.7288818, 555.7738037, -755.0511475, 888.9410400

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5787930, upper bound: 560.5727776
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5860222, upper bound: 560.5890206
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5871431, upper bound: 560.5895506
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -182.5155792, 572.6014404, -179.3179016, 552.2359009, -734.7514038, 751.9193115
1: -259.6409302, 577.7786865, -254.8494263, 559.2086792, -818.8496094, 832.6279907
2: -219.5419617, 638.3512573, -215.5402527, 618.3243408, -837.8662109, 853.8914795
3: -234.1709137, 803.0667114, -229.6629944, 775.3743896, -1009.5451660, 1032.7296143
4: -196.8828735, 739.0787354, -192.9681549, 715.3275757, -912.2104492, 932.0466309

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -180.8464661, 567.3162231, -280.7216187, 868.0888672, -1046.4774170, 848.0378418
1: -257.2435608, 572.4565430, -398.7448730, 877.7628174, -1132.0606689, 970.7420654
2: -217.5253601, 632.4747925, -336.5735168, 970.3506470, -1184.1197510, 968.8282471
3: -232.0157166, 795.6259155, -359.4641418, 1214.0162354, -1443.8052979, 1155.0284424
4: -195.0777588, 732.2266846, -301.7256775, 1118.9639893, -1312.6917725, 1033.3676758

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -185.2178802, 569.7943726, -180.8099976, 556.1011963, -741.3190308, 750.6043701
1: -263.1491394, 577.0369263, -256.9301758, 563.2346191, -826.3836670, 833.9671021
2: -222.5697784, 637.9903564, -217.2973633, 622.8032227, -845.3729858, 855.2876587
3: -237.1614532, 800.1052856, -231.5204010, 780.8549194, -1018.0163574, 1031.6257324
4: -199.2773132, 738.2121582, -194.5097809, 720.4786377, -919.7559814, 932.7219238

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -183.7976990, 565.2296753, -282.3350525, 872.2695312, -1053.6914062, 847.5646973
1: -261.1152649, 572.4295654, -400.9938660, 882.0807495, -1140.2862549, 973.0836182
2: -220.8601532, 632.8949585, -338.4806824, 975.1448364, -1192.3140869, 971.2363892
3: -235.3276062, 793.6861572, -361.4702759, 1219.9283447, -1453.0524902, 1155.1564941
4: -197.7467957, 732.3082275, -303.4025574, 1124.5421143, -1320.9674072, 1035.3159180

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.55 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5797149, upper bound: 560.5674417
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5797149, upper bound: 560.5789173
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5748618
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5793880, upper bound: 560.5688676
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5707075, upper bound: 560.5668220
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5793880, upper bound: 560.5728390
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5707075, upper bound: 560.5710801
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5766031, upper bound: 560.5737768
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5738353
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5766031, upper bound: 560.5798900
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5799485
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5823342
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5831281
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5869437
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5893473, upper bound: 560.5876440
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5728483, upper bound: 560.5815241
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5824776, upper bound: 560.5839254
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5728483, upper bound: 560.5869460
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5824776, upper bound: 560.5893473
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5860222, upper bound: 560.5835987
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5871431, upper bound: 560.5843363
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5860222, upper bound: 560.5890206
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5871431, upper bound: 560.5895506
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -560.5563203, upper bound: 560.5567068

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -132.6752014, 407.8443604, -534.8416138, 522.3602905
1: -180.3264771, 395.6703491, -188.4647369, 414.0709229, -594.3973389, 584.1350708
2: -152.4571991, 437.5143433, -159.2985992, 457.8714600, -610.3286743, 596.8129272
3: -162.3547211, 549.1265259, -169.7159882, 574.7191162, -737.0737915, 718.8425293
4: -136.4553528, 506.8262329, -142.6433258, 530.3728638, -666.8282471, 649.4695435

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -132.3054810, 406.6438293, -568.6876221, 630.9702148
1: -229.8305664, 505.0910034, -187.9435425, 412.8746033, -642.7050781, 693.0345459
2: -194.1427002, 558.3701172, -158.8598785, 456.5573730, -650.7000732, 717.2299805
3: -207.1919708, 699.0578003, -169.2457275, 573.0498047, -780.2417603, 868.3035278
4: -174.0971069, 645.9729614, -142.2519989, 528.8516846, -702.9487305, 788.2249146

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -140.9674835, 430.3715820, -557.3688965, 530.6526489
1: -180.3264771, 395.6703491, -199.8453827, 436.4012756, -616.7277832, 595.5157471
2: -152.4571991, 437.5143433, -168.9626923, 481.6351929, -634.0924072, 606.4770508
3: -162.3547211, 549.1265259, -179.8964081, 603.5338135, -765.8885498, 729.0229492
4: -136.4553528, 506.8262329, -150.7288818, 555.7738037, -692.2291260, 657.5550537

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5643546, upper bound: 560.5657123
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5663038, upper bound: 560.5686447
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -140.6368866, 429.3018188, -591.3456421, 639.3016968
1: -229.8305664, 505.0910034, -199.3748474, 435.3339844, -665.1645508, 704.4658203
2: -194.1427002, 558.3701172, -168.5673676, 480.4660645, -674.6087646, 726.9374390
3: -207.1919708, 699.0578003, -179.4722290, 602.0479126, -809.2398682, 878.5300293
4: -174.0971069, 645.9729614, -150.3750305, 554.4142456, -728.5113525, 796.3480225

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5602974, upper bound: 560.5644081
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5622467, upper bound: 560.5673405
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -136.6867371, 417.2817078, -132.6752014, 407.8443604, -544.5311279, 549.9567261
1: -194.0063324, 423.1643982, -188.4647369, 414.0709229, -608.0772705, 611.6291504
2: -164.0225830, 467.0018921, -159.2985992, 457.8714600, -621.8940430, 626.3004761
3: -174.5901337, 584.9263916, -169.7159882, 574.7191162, -749.3092041, 754.6423340
4: -146.2916565, 538.6610107, -142.6433258, 530.3728638, -676.6644897, 681.3043213

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5657123, upper bound: 560.5643546
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5644081, upper bound: 560.5602974
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -145.2435303, 447.6614990, -132.3549042, 406.9267578, -552.1701050, 580.0162964
1: -205.9735260, 452.7306824, -187.9926147, 413.1221008, -619.0955811, 640.7232666
2: -174.0315247, 499.1432495, -158.9012299, 456.8204956, -630.8519287, 658.0444946
3: -185.4028473, 626.1638184, -169.2803497, 573.4238281, -758.8266602, 795.4441528
4: -155.2832336, 575.0490723, -142.3059387, 529.1372070, -684.4204102, 717.3549194

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5686447, upper bound: 560.5663038
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5673405, upper bound: 560.5622467
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -136.6867371, 417.2817078, -140.9674835, 430.3715820, -567.0583496, 558.2491455
1: -194.0063324, 423.1643982, -199.8453827, 436.4012756, -630.4075928, 623.0097656
2: -164.0225830, 467.0018921, -168.9626923, 481.6351929, -645.6577148, 635.9645386
3: -174.5901337, 584.9263916, -179.8964081, 603.5338135, -778.1239624, 764.8227539
4: -146.2916565, 538.6610107, -150.7288818, 555.7738037, -702.0654297, 689.3898315

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5711188, upper bound: 560.5710801
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5711188, upper bound: 560.5710801
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -145.2435303, 447.6614990, -140.6611023, 429.5030212, -574.7463989, 588.3225708
1: -205.9735260, 452.7306824, -199.3979797, 435.5057068, -641.4790649, 652.1285400
2: -174.0315247, 499.1432495, -168.5853424, 480.6401672, -654.6714478, 667.7285767
3: -185.4028473, 626.1638184, -179.4818115, 602.3088989, -787.7117310, 805.6455688
4: -155.2832336, 575.0490723, -150.3941956, 554.6063232, -709.8895264, 725.4432373

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5690560, upper bound: 560.5706006
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684974, upper bound: 560.5685379
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -182.5114288, 572.5888062, -693.9846191, 556.8289795
1: -172.1784973, 379.8592529, -259.6349792, 577.7658081, -749.9442749, 639.4942017
2: -145.6561737, 420.1729736, -219.5369568, 638.3371582, -783.9933472, 639.7098999
3: -155.1810760, 527.8892212, -234.1655273, 803.0490723, -958.2301636, 762.0545654
4: -130.6282196, 486.8346863, -196.8783875, 739.0623169, -869.6905518, 683.7130127

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651240, upper bound: 560.5666154
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637891, upper bound: 560.5638438
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5686676, upper bound: 560.5195971
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5766031, upper bound: 560.5737768
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -125.5253906, 384.8227234, -182.5155792, 572.6014404, -698.1268311, 567.3382568
1: -178.2565460, 390.9218750, -259.6409302, 577.7786865, -756.0351562, 650.5628052
2: -150.7256012, 432.3974304, -219.5419617, 638.3512573, -789.0768433, 651.9392090
3: -160.4758301, 542.5620117, -234.1709137, 803.0667114, -963.5425415, 776.7328491
4: -134.9489594, 500.8383789, -196.8828735, 739.0787354, -874.0277100, 697.7212524

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5665792, upper bound: 560.5196556
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5738353
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -185.2140350, 569.7830200, -691.1788330, 559.5315552
1: -172.1784973, 379.8592529, -263.1435547, 577.0253906, -749.2038574, 643.0026855
2: -145.6561737, 420.1729736, -222.5650940, 637.9776001, -783.6337891, 642.7380371
3: -155.1810760, 527.8892212, -237.1564331, 800.0893555, -955.2704468, 765.0455933
4: -130.6282196, 486.8346863, -199.2731476, 738.1973267, -868.8255615, 686.1078491

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5654993, upper bound: 560.5722746
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5769308, upper bound: 560.5775534
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5775848, upper bound: 560.5792856
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -125.5253906, 384.8227234, -185.2178802, 569.7943726, -695.3197632, 570.0405884
1: -178.2565460, 390.9218750, -263.1491394, 577.0369263, -755.2934570, 654.0709229
2: -150.7256012, 432.3974304, -222.5697784, 637.9903564, -788.7159424, 654.9672241
3: -160.4758301, 542.5620117, -237.1614532, 800.1052856, -960.5811157, 779.7234497
4: -134.9489594, 500.8383789, -199.2773132, 738.2121582, -873.1611328, 700.1157227

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5748423, upper bound: 560.5776119
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5793441
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.0930405, 373.0067749, -182.5155792, 572.6014404, -695.6944580, 555.5222778
1: -174.2050018, 379.3984070, -259.6409302, 577.7786865, -751.9835205, 639.0393066
2: -147.2826996, 419.4069824, -219.5419617, 638.3512573, -785.6339111, 638.9488525
3: -157.0052032, 524.2232666, -234.1709137, 803.0667114, -960.0718994, 758.3940430
4: -131.4749298, 483.7127380, -196.8828735, 739.0787354, -870.5536499, 680.5955811

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5547403, upper bound: 560.5773509
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5727049
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5823342
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.0707550, 418.1671753, -182.5155792, 572.6014404, -709.6721802, 600.6826782
1: -194.3686218, 424.1262207, -259.6409302, 577.7786865, -772.1472168, 683.7671509
2: -164.3190765, 468.1112061, -219.5419617, 638.3512573, -802.6703491, 687.6531372
3: -174.9659729, 586.4363403, -234.1709137, 803.0667114, -978.0325928, 820.6071167
4: -146.5706940, 540.0454712, -196.8828735, 739.0787354, -885.6494141, 736.9283447

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5380401, upper bound: 560.5516645
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5734988
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5831281
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.0930405, 373.0067749, -185.2178802, 569.7943726, -692.8873901, 558.2246704
1: -174.2050018, 379.3984070, -263.1491394, 577.0369263, -751.2418823, 642.5474854
2: -147.2826996, 419.4069824, -222.5697784, 637.9903564, -785.2730103, 641.9767456
3: -157.0052032, 524.2232666, -237.1614532, 800.1052856, -957.1104736, 761.3847046
4: -131.4749298, 483.7127380, -199.2773132, 738.2121582, -869.6870728, 682.9900513

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5551385, upper bound: 560.5812478
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5731490, upper bound: 560.5791513
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885654, upper bound: 560.5858177
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5869387
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -137.0707550, 418.1671753, -185.2178802, 569.7943726, -706.8650513, 603.3850098
1: -194.3686218, 424.1262207, -263.1491394, 577.0369263, -771.4055176, 687.2750854
2: -164.3190765, 468.1112061, -222.5697784, 637.9903564, -802.3094482, 690.6809692
3: -174.9659729, 586.4363403, -237.1614532, 800.1052856, -975.0711670, 823.5977783
4: -146.5706940, 540.0454712, -199.2773132, 738.2121582, -884.7828369, 739.3227539

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5419860, upper bound: 560.5760425
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885654, upper bound: 560.5863097
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5876440
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -166.0689087, 518.8889771, -132.6752014, 407.8443604, -573.9132690, 651.5641479
1: -236.4929962, 524.5581665, -188.4647369, 414.0709229, -650.5639038, 713.0228882
2: -200.0235138, 579.8539429, -159.2985992, 457.8714600, -657.8949585, 739.1524658
3: -213.1714325, 728.6763916, -169.7159882, 574.7191162, -787.8905640, 898.3923340
4: -179.3784027, 671.0729370, -142.6433258, 530.3728638, -709.7511597, 813.7162476

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5072373, upper bound: 560.5734543
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5493299, upper bound: 560.5709036
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718766, upper bound: 560.5805029
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5718766, upper bound: 560.5815241
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -173.8404846, 545.8111572, -132.6752014, 407.8443604, -581.6847534, 678.4862671
1: -247.1605682, 550.6942139, -188.4647369, 414.0709229, -661.2314453, 739.1589355
2: -209.0529785, 608.3787231, -159.2985992, 457.8714600, -666.9244385, 767.6772461
3: -223.0072479, 765.3759155, -169.7159882, 574.7191162, -797.7262573, 935.0918579
4: -187.4968109, 704.5477295, -142.6433258, 530.3728638, -717.8696289, 847.1910400

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5565082, upper bound: 560.5804530
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5735399, upper bound: 560.5765110
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5736265, upper bound: 560.5745146
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -166.0689087, 518.8889771, -140.9674835, 430.3715820, -596.4404907, 659.8564453
1: -236.4929962, 524.5581665, -199.8453827, 436.4012756, -672.8942871, 724.4035645
2: -200.0235138, 579.8539429, -168.9626923, 481.6351929, -681.6586304, 748.8165283
3: -213.1714325, 728.6763916, -179.8964081, 603.5338135, -816.7052612, 908.5727539
4: -179.3784027, 671.0729370, -150.7288818, 555.7738037, -735.1522217, 821.8017578

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5078878, upper bound: 560.5788762
time: 1.53 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5496612, upper bound: 560.5765186
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5727049, upper bound: 560.5869460
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5727049, upper bound: 560.5869460
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -173.8404846, 545.8111572, -140.9674835, 430.3715820, -604.2120361, 686.7786255
1: -247.1605682, 550.6942139, -199.8453827, 436.4012756, -683.5618286, 750.5396118
2: -209.0529785, 608.3787231, -168.9626923, 481.6351929, -690.6881104, 777.3413086
3: -223.0072479, 765.3759155, -179.8964081, 603.5338135, -826.5410156, 945.2722168
4: -187.4968109, 704.5477295, -150.7288818, 555.7738037, -743.2706299, 855.2766113

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5571587, upper bound: 560.5858749
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815059, upper bound: 560.5893473
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5815059, upper bound: 560.5893473
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -165.3141327, 506.1864319, -132.6296387, 407.7038879, -573.0180054, 638.8160400
1: -233.7621460, 512.2211914, -188.4016724, 413.9297485, -647.6918335, 700.6228638
2: -197.6934204, 566.1394043, -159.2455750, 457.7153015, -655.4086914, 725.3849487
3: -210.9736786, 709.6657715, -169.6592865, 574.5226440, -785.4960327, 879.3250732
4: -177.0330200, 654.5571899, -142.5958405, 530.1913452, -707.2243652, 797.1529541

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574663, upper bound: 560.5791610
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5775534, upper bound: 560.5769308
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5776119, upper bound: 560.5748423
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -179.8043976, 554.5242920, -132.6752014, 407.8443604, -587.6486816, 687.1992798
1: -255.7558441, 561.4832764, -188.4647369, 414.0709229, -669.8266602, 749.9479980
2: -216.3901520, 620.7167358, -159.2985992, 457.8714600, -674.2615967, 780.0152588
3: -230.5191650, 778.6634521, -169.7159882, 574.7191162, -805.2382812, 948.3794556
4: -193.7203674, 718.0985107, -142.6433258, 530.3728638, -724.0932617, 860.7418213

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5600241, upper bound: 560.5804530
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792856, upper bound: 560.5775848
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5793441, upper bound: 560.5754963
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -165.3141327, 506.1864319, -140.9369354, 430.2714539, -595.5855103, 647.1233521
1: -233.7621460, 512.2211914, -199.8018341, 436.3019409, -670.0640259, 712.0230103
2: -197.6934204, 566.1394043, -168.9263458, 481.5259094, -679.2193604, 735.0657349
3: -210.9736786, 709.6657715, -179.8573914, 603.3945923, -814.3679810, 889.5231934
4: -177.0330200, 654.5571899, -150.6964569, 555.6463623, -732.6793823, 805.2535400

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5580558, upper bound: 560.5845829
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5858046, upper bound: 560.5890206
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866117, upper bound: 560.5890206
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -179.8043976, 554.5242920, -140.9674835, 430.3715820, -610.1759644, 695.4916382
1: -255.7558441, 561.4832764, -199.8453827, 436.4012756, -692.1571045, 761.3286743
2: -216.3901520, 620.7167358, -168.9626923, 481.6351929, -698.0253296, 789.6793213
3: -230.5191650, 778.6634521, -179.8964081, 603.5338135, -834.0529785, 958.5598145
4: -193.7203674, 718.0985107, -150.7288818, 555.7738037, -749.4941406, 868.8273926

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5606103, upper bound: 560.5851112
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867849, upper bound: 560.5895506
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5877326, upper bound: 560.5895506
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -179.3179016, 552.2359009, -727.5879517, 729.8497314
1: -249.4921265, 555.4519653, -254.8494263, 559.2086792, -808.7007446, 810.3013306
2: -210.9501648, 613.7087402, -215.5402527, 618.3243408, -829.2744751, 829.2490234
3: -224.9954834, 771.8947754, -229.6629944, 775.3743896, -1000.3698730, 1001.5577393
4: -189.1553650, 710.2475586, -192.9681549, 715.3275757, -904.4829102, 903.2155151

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5792213
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5838308
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -179.3179016, 552.2359009, -828.2478027, 1044.0786133
1: -392.7666626, 875.2170410, -254.8494263, 559.2086792, -951.5950317, 1126.6644287
2: -331.3969727, 966.5505371, -215.5402527, 618.3243408, -949.4953003, 1178.0449219
3: -354.2966614, 1211.7148438, -229.6629944, 775.3743896, -1129.6710205, 1438.0629883
4: -297.2891541, 1113.6004639, -192.9681549, 715.3275757, -1012.2731934, 1304.7579346

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5792213
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5838308
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -280.7216187, 868.0888672, -1040.9465332, 831.2534180
1: -249.4921265, 555.4519653, -398.7448730, 877.7628174, -1124.2579346, 953.6663208
2: -210.9501648, 613.7087402, -336.5735168, 970.3506470, -1177.4924316, 949.9783325
3: -224.9954834, 771.8947754, -359.4641418, 1214.0162354, -1436.7406006, 1131.2530518
4: -189.1553650, 710.2475586, -301.7256775, 1118.9639893, -1306.7423096, 1011.3621826

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5532519
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -280.7216187, 868.0888672, -1140.6196289, 1144.4438477
1: -392.7666626, 875.2170410, -398.7448730, 877.7628174, -1265.7932129, 1268.7510986
2: -331.3969727, 966.5505371, -336.5735168, 970.3506470, -1296.4661865, 1297.5449219
3: -354.2966614, 1211.7148438, -359.4641418, 1214.0162354, -1564.6484375, 1566.4395752
4: -297.2891541, 1113.6004639, -301.7256775, 1118.9639893, -1413.3437500, 1411.9626465

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5532519
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -180.8099976, 556.1011963, -733.6386108, 726.5734863
1: -252.2587128, 552.8828125, -256.9301758, 563.2346191, -815.4933472, 809.8129272
2: -213.3510437, 611.4344482, -217.2973633, 622.8032227, -836.1540527, 828.7317505
3: -227.3259888, 766.4622803, -231.5204010, 780.8549194, -1008.1809082, 997.9826050
4: -191.0043335, 707.2424316, -194.5097809, 720.4786377, -911.4829712, 901.7521973

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5792213
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5830001
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -180.8099976, 556.1011963, -836.0996704, 1042.8066406
1: -397.6463318, 874.5075073, -256.9301758, 563.2346191, -960.4857178, 1128.3966064
2: -335.6636353, 966.8534546, -217.2973633, 622.8032227, -958.2635498, 1180.3336182
3: -358.4500122, 1209.4127197, -231.5204010, 780.8549194, -1139.3049316, 1438.3845215
4: -300.8917236, 1114.8928223, -194.5097809, 720.4786377, -1021.0316772, 1307.8937988

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5792213
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5830001
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -282.3350525, 872.2695312, -1047.4051514, 828.0985107
1: -252.2587128, 552.8828125, -400.9938660, 882.0807495, -1131.3818359, 953.4799194
2: -213.3510437, 611.4344482, -338.4806824, 975.1448364, -1184.7579346, 949.7021484
3: -227.3259888, 766.4622803, -361.4702759, 1219.9283447, -1445.0112305, 1127.9326172
4: -191.0043335, 707.2424316, -303.4025574, 1124.5421143, -1314.2087402, 1010.2222290

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5532519
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5563203
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -282.3350525, 872.2695312, -1148.8226318, 1143.2962646
1: -397.6463318, 874.5075073, -400.9938660, 882.0807495, -1274.9671631, 1270.6561279
2: -335.6636353, 966.8534546, -338.4806824, 975.1448364, -1305.5426025, 1299.9868164
3: -358.4500122, 1209.4127197, -361.4702759, 1219.9283447, -1574.7038574, 1566.9151611
4: -300.8917236, 1114.8928223, -303.4025574, 1124.5421143, -1422.5340576, 1415.2404785

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5532519
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5563203
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.83 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5633863, upper bound: 560.5633863
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5643546, upper bound: 560.5657123
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5663038, upper bound: 560.5686447
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5602974, upper bound: 560.5644081
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5622467, upper bound: 560.5673405
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5657123, upper bound: 560.5643546
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5644081, upper bound: 560.5602974
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5686447, upper bound: 560.5663038
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5673405, upper bound: 560.5622467
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5711188, upper bound: 560.5710801
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5711188, upper bound: 560.5710801
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5690560, upper bound: 560.5706006
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5684974, upper bound: 560.5685379
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5686676, upper bound: 560.5195971
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5766031, upper bound: 560.5737768
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5665792, upper bound: 560.5196556
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5738353
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5769308, upper bound: 560.5775534
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5775848, upper bound: 560.5792856
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5748423, upper bound: 560.5776119
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5745146, upper bound: 560.5793441
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5727049
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5823342
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5734988
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5869460, upper bound: 560.5831281
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5885654, upper bound: 560.5858177
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5869387
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5885654, upper bound: 560.5863097
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5897582, upper bound: 560.5876440
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5718766, upper bound: 560.5805029
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5718766, upper bound: 560.5815241
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5735399, upper bound: 560.5765110
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5736265, upper bound: 560.5745146
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5727049, upper bound: 560.5869460
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5727049, upper bound: 560.5869460
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5815059, upper bound: 560.5893473
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5815059, upper bound: 560.5893473
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5775534, upper bound: 560.5769308
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5776119, upper bound: 560.5748423
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5792856, upper bound: 560.5775848
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5793441, upper bound: 560.5754963
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5858046, upper bound: 560.5890206
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5866117, upper bound: 560.5890206
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5867849, upper bound: 560.5895506
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5877326, upper bound: 560.5895506
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5792213
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5838308
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5792213
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5567243, upper bound: 560.5838308
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5532519
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5532519
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5532519, upper bound: 560.5567068
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5792213
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5830001
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5792213
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5601792, upper bound: 560.5830001
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5532519
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5563203
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5532519
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.83
Output dim: 0, lower bound: -560.5555956, upper bound: 560.5563203

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -126.9973297, 389.6851501, -516.6824341, 516.6823730
1: -180.3264771, 395.6703491, -180.3264771, 395.6703491, -575.9968262, 575.9968262
2: -152.4571991, 437.5143433, -152.4571991, 437.5143433, -589.9714966, 589.9715576
3: -162.3547211, 549.1265259, -162.3547211, 549.1265259, -711.4812622, 711.4812622
4: -136.4553528, 506.8262329, -136.4553528, 506.8262329, -643.2816162, 643.2816162

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5724688, upper bound: 560.5644791
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651065, upper bound: 560.5623535
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -162.0438080, 498.6647949, -625.6619873, 551.7289429
1: -180.3264771, 395.6703491, -229.8305664, 505.0910034, -685.4174805, 625.5008545
2: -152.4571991, 437.5143433, -194.1427002, 558.3701172, -710.8273315, 631.6570435
3: -162.3547211, 549.1265259, -207.1919708, 699.0578003, -861.4125366, 756.3184814
4: -136.4553528, 506.8262329, -174.0971069, 645.9729614, -782.4283447, 680.9232178

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5724688, upper bound: 560.5644791
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651065, upper bound: 560.5623535
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -126.9973297, 389.6851501, -551.7289429, 625.6619873
1: -229.8305664, 505.0910034, -180.3264771, 395.6703491, -625.5008545, 685.4174805
2: -194.1427002, 558.3701172, -152.4571991, 437.5143433, -631.6570435, 710.8273315
3: -207.1919708, 699.0578003, -162.3547211, 549.1265259, -756.3184814, 861.4125366
4: -174.0971069, 645.9729614, -136.4553528, 506.8262329, -680.9232788, 782.4283447

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626049, upper bound: 560.5616723
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5610675, upper bound: 560.5610675
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -162.0438080, 498.6647949, -660.7084961, 660.7084961
1: -229.8305664, 505.0910034, -229.8305664, 505.0910034, -734.9215698, 734.9215698
2: -194.1427002, 558.3701172, -194.1427002, 558.3701172, -752.5128174, 752.5128174
3: -207.1919708, 699.0578003, -207.1919708, 699.0578003, -906.2497559, 906.2497559
4: -174.0971069, 645.9729614, -174.0971069, 645.9729614, -820.0700684, 820.0700073

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626049, upper bound: 560.5616723
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5610675, upper bound: 560.5610675
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -136.6867371, 417.2817078, -544.2788696, 526.3718872
1: -180.3264771, 395.6703491, -194.0063324, 423.1643982, -603.4908447, 589.6766968
2: -152.4571991, 437.5143433, -164.0225830, 467.0018921, -619.4591064, 601.5368652
3: -162.3547211, 549.1265259, -174.5901337, 584.9263916, -747.2811279, 723.7166748
4: -136.4553528, 506.8262329, -146.2916565, 538.6610107, -675.1163330, 653.1177979

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5643546, upper bound: 560.5656513
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5643546, upper bound: 560.5656513
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -126.6607742, 388.7520447, -145.2435303, 447.6614990, -574.3222046, 533.9954224
1: -179.8484344, 394.7084961, -205.9735260, 452.7306824, -632.5791016, 600.6818237
2: -152.0536194, 436.4499207, -174.0315247, 499.1432495, -651.1968994, 610.4812622
3: -161.9120178, 547.8128662, -185.4028473, 626.1638184, -788.0758057, 733.2156982
4: -136.1080322, 505.5739441, -155.2832336, 575.0490723, -711.1571045, 660.8571777

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5663038, upper bound: 560.5686447
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5663038, upper bound: 560.5686447
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -136.3429108, 416.1712341, -578.2150269, 635.0076294
1: -229.8305664, 505.0910034, -193.5167694, 422.0562134, -651.8865967, 698.6077881
2: -194.1427002, 558.3701172, -163.6109009, 465.7884521, -659.9311523, 721.9810181
3: -207.1919708, 699.0578003, -174.1487579, 583.3837891, -790.5757446, 873.2065430
4: -174.0971069, 645.9729614, -145.9233551, 537.2490845, -711.3461914, 791.8963013

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5550928, upper bound: 560.5398586
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5570975, upper bound: 560.5604287
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5568258, upper bound: 560.5525200
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5576896, upper bound: 560.5638409
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5573999, upper bound: 560.5609531
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5525119, upper bound: 560.5500014
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8

Time for candidate selection: 13.69 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5535377, upper bound: 560.5615893
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5506131, upper bound: 560.5607124
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -161.4212341, 496.8800049, -144.8479004, 446.3861694, -607.8073730, 641.7279053
1: -228.9889984, 503.2577515, -205.4114838, 451.4563599, -680.4453125, 708.6692505
2: -193.4246521, 556.3244019, -173.5580902, 497.7456665, -691.1701660, 729.8825073
3: -206.4132080, 696.5048828, -184.8969574, 624.3864136, -830.7994995, 881.4017944
4: -173.4370117, 643.5686646, -154.8612061, 573.4213867, -746.8583374, 798.4296875

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5597801, upper bound: 560.5444068
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557627, upper bound: 560.5466111
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5556030, upper bound: 560.5527405
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5615146, upper bound: 560.5656326
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5598560, upper bound: 560.5649943
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -136.6867371, 417.2817078, -126.9973297, 389.6851501, -526.3718872, 544.2788696
1: -194.0063324, 423.1643982, -180.3264771, 395.6703491, -589.6766968, 603.4908447
2: -164.0225830, 467.0018921, -152.4571991, 437.5143433, -601.5369263, 619.4591064
3: -174.5901337, 584.9263916, -162.3547211, 549.1265259, -723.7166748, 747.2811279
4: -146.2916565, 538.6610107, -136.4553528, 506.8262329, -653.1177979, 675.1163330

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5644081, upper bound: 560.5602974
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5644081, upper bound: 560.5602974
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -136.3429108, 416.1712341, -162.0438080, 498.6647949, -635.0076294, 578.2150269
1: -193.5167694, 422.0562134, -229.8305664, 505.0910034, -698.6077881, 651.8866577
2: -163.6109009, 465.7884521, -194.1427002, 558.3701172, -721.9810181, 659.9311523
3: -174.1487579, 583.3837891, -207.1919708, 699.0578003, -873.2065430, 790.5757446
4: -145.9233551, 537.2490845, -174.0971069, 645.9729614, -791.8963013, 711.3461914

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5644081, upper bound: 560.5602974
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5644081, upper bound: 560.5602974
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -145.2435303, 447.6614990, -126.6607742, 388.7520447, -533.9954224, 574.3222046
1: -205.9735260, 452.7306824, -179.8484344, 394.7084961, -600.6818848, 632.5791016
2: -174.0315247, 499.1432495, -152.0536194, 436.4499207, -610.4813232, 651.1968994
3: -185.4028473, 626.1638184, -161.9120178, 547.8128662, -733.2156982, 788.0758057
4: -155.2832336, 575.0490723, -136.1080322, 505.5739441, -660.8571777, 711.1571045

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5592651, upper bound: 560.5574320
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5673405, upper bound: 560.5622467
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5673405, upper bound: 560.5622467
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -144.8479004, 446.3861694, -161.4212341, 496.8800049, -641.7279053, 607.8073730
1: -205.4114838, 451.4563599, -228.9889984, 503.2577515, -708.6692505, 680.4453125
2: -173.5580902, 497.7456665, -193.4246521, 556.3244019, -729.8825073, 691.1701660
3: -184.8969574, 624.3864136, -206.4132080, 696.5048828, -881.4017944, 830.7994385
4: -154.8612061, 573.4213867, -173.4370117, 643.5686646, -798.4296875, 746.8583374

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5007875, upper bound: 560.5497836
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5673405, upper bound: 560.5622467
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -136.6867371, 417.2817078, -136.6867371, 417.2817078, -553.9684448, 553.9684448
1: -194.0063324, 423.1643982, -194.0063324, 423.1643982, -617.1707153, 617.1707153
2: -164.0225830, 467.0018921, -164.0225830, 467.0018921, -631.0244751, 631.0244751
3: -174.5901337, 584.9263916, -174.5901337, 584.9263916, -759.5165405, 759.5165405
4: -146.2916565, 538.6610107, -146.2916565, 538.6610107, -684.9525757, 684.9525757

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726703, upper bound: 560.5650401
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5636562, upper bound: 560.5635952
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -136.6867371, 417.2817078, -145.2435303, 447.6614990, -584.3482666, 562.5250244
1: -194.0063324, 423.1643982, -205.9735260, 452.7306824, -646.7369995, 629.1379395
2: -164.0225830, 467.0018921, -174.0315247, 499.1432495, -663.1658325, 641.0333252
3: -174.5901337, 584.9263916, -185.4028473, 626.1638184, -800.7539673, 770.3292236
4: -146.2916565, 538.6610107, -155.2832336, 575.0490723, -721.3406982, 693.9441528

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726703, upper bound: 560.5707111
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5636562, upper bound: 560.5665886
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -145.2435303, 447.6614990, -134.6863556, 410.6206665, -555.8640747, 582.3478394
1: -205.9735260, 452.7306824, -190.9188080, 416.2895813, -622.2629395, 643.6494751
2: -174.0315247, 499.1432495, -161.4398193, 459.3284302, -633.3598022, 660.5830688
3: -185.4028473, 626.1638184, -171.8042450, 575.6234741, -761.0263062, 797.9680176
4: -155.2832336, 575.0490723, -143.9416809, 530.0681152, -685.3513184, 718.9907227

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5594868, upper bound: 560.5578887
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684974, upper bound: 560.5685379
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684974, upper bound: 560.5685379
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.8479004, 446.3861694, -169.6849060, 525.7630005, -670.6109009, 616.0709839
1: -205.4114838, 451.4563599, -241.5823059, 531.4892578, -736.9006958, 693.0385132
2: -173.5580902, 497.7456665, -204.0070343, 586.5338745, -760.0918579, 701.7526855
3: -184.8969574, 624.3864136, -217.6072998, 734.4997559, -919.3966675, 841.9935913
4: -154.8612061, 573.4213867, -182.3307953, 676.1697998, -831.0309448, 755.7521362

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5019849, upper bound: 560.5560748
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5684974, upper bound: 560.5685379
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -121.0778351, 373.3037415, -191.9763794, 600.5520630, -721.6298828, 565.2800293
1: -171.7255859, 378.8367920, -273.3803101, 606.1790771, -777.9045410, 652.2171021
2: -145.2731323, 419.0469666, -231.1002808, 669.4301758, -814.7033081, 650.1470947
3: -154.7731781, 526.4721680, -246.5080414, 841.4132690, -996.1862793, 772.9801636
4: -130.2850952, 485.5366516, -207.0496368, 774.5676270, -904.8527222, 692.5862427

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5686676, upper bound: 560.5195971
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5659172, upper bound: 560.5186640
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5685755, upper bound: 560.5193602
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -182.1303406, 571.4237671, -692.8195801, 556.4478760
1: -172.1784973, 379.8592529, -259.0946960, 576.5830688, -748.7615967, 638.9539795
2: -145.6561737, 420.1729736, -219.0777893, 637.0336304, -782.6898193, 639.2507324
3: -155.1810760, 527.8892212, -233.6787262, 801.4044800, -956.5855713, 761.5679321
4: -130.6282196, 486.8346863, -196.4692535, 737.5444946, -868.1727295, 683.3038330

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5737145, upper bound: 560.5456219
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5738527, upper bound: 560.5728438
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5765110, upper bound: 560.5735399
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -125.1928787, 383.7779541, -191.9803162, 600.5642700, -725.7570801, 575.7583008
1: -177.7852020, 389.8668518, -273.3860779, 606.1914673, -783.9766235, 663.2529297
2: -150.3265533, 431.2337646, -231.1051025, 669.4436646, -819.7702026, 662.3388672
3: -160.0511017, 541.0956421, -246.5131989, 841.4301147, -1001.4811401, 787.6088257
4: -134.5910339, 499.4908752, -207.0539246, 774.5833740, -909.1744385, 706.5447998

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5665792, upper bound: 560.5196556
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=618.8850708007812
rel_dist={0: [-560.590385842507, 560.590385842507]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

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
- Time for IS candidates: 2.18 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -560.5900352, upper bound: 560.5880940
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -151.0581512, 461.7381592, -608.3136597, 598.4097900
1: -207.9424896, 453.5178833, -214.3221588, 467.8475647, -675.7900391, 667.8400269
2: -175.7731476, 500.5854492, -181.1492157, 516.3231812, -692.0963135, 681.7346802
3: -187.1182709, 627.4591675, -192.8606873, 647.4821167, -834.6004028, 820.3197632
4: -156.8196411, 578.2070312, -161.6295776, 596.5396118, -753.3591919, 739.8366089

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 1.00 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
time: 0.96 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -149.5736237, 457.5722046, -646.1802979, 730.0730591
1: -267.9796143, 587.7565918, -212.2177429, 463.4609375, -731.4405518, 799.9742432
2: -226.6574402, 649.7827759, -179.3684692, 511.4507446, -738.1081543, 829.1511841
3: -241.4969330, 815.0511475, -190.9532623, 641.4743042, -882.9712524, 1006.0043945
4: -202.9143524, 751.9616089, -160.0287476, 590.9093628, -793.8236694, 911.9903564

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 1.13 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 1.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.94 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 0, lower bound: -560.5880810, upper bound: 560.5880810
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.94
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -146.5755005, 447.3516541, -593.9271240, 593.9270630
1: -207.9424896, 453.5178833, -207.9424896, 453.5178833, -661.4603882, 661.4603882
2: -175.7731476, 500.5854492, -175.7731476, 500.5854492, -676.3585815, 676.3585815
3: -187.1182709, 627.4591675, -187.1182709, 627.4591675, -814.5774536, 814.5774536
4: -156.8196411, 578.2070312, -156.8196411, 578.2070312, -735.0266724, 735.0266724

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5839872, upper bound: 560.5870092
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5875312
time: 0.94 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -188.6081696, 580.4994507, -727.0748901, 635.9598389
1: -207.9424896, 453.5178833, -267.9796143, 587.7565918, -795.6990967, 721.4974976
2: -175.7731476, 500.5854492, -226.6574402, 649.7827759, -825.5559082, 727.2428589
3: -187.1182709, 627.4591675, -241.4969330, 815.0511475, -1002.1694336, 868.9561157
4: -156.8196411, 578.2070312, -202.9143524, 751.9616089, -908.7811890, 781.1213989

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5839872, upper bound: 560.5870092
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5875312
time: 0.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -188.1006927, 578.9155884, -143.7614288, 438.8794250, -626.9801025, 722.6770020
1: -267.2606506, 586.1631470, -203.9528351, 444.7846375, -712.0452881, 790.1159668
2: -226.0482635, 648.0296631, -172.3946381, 490.9169922, -716.9652710, 820.4242554
3: -240.8472900, 812.8301392, -183.5071411, 615.3638306, -856.2110596, 996.3372192
4: -202.3665466, 749.9147949, -153.7962036, 567.0385742, -769.4051514, 903.7109985

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 1.11 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -184.4823456, 567.1229858, -204.9295197, 625.3498535, -809.8322144, 772.0524902
1: -262.0635071, 574.2723999, -290.3470154, 633.9213257, -895.7351074, 864.6193848
2: -221.6806030, 634.8838501, -245.0624542, 700.8681030, -921.8262939, 879.9462280
3: -236.1643524, 796.2614136, -261.6196899, 875.7595825, -1111.9239502, 1057.8809814
4: -198.4624176, 734.7160034, -219.3437500, 808.1102295, -1006.5725708, 954.0597534

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.40 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5839872, upper bound: 560.5870092
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5875312
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5839872, upper bound: 560.5870092
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5875312
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -141.8082428, 435.1930847, -567.8682861, 549.6525879
1: -188.4647369, 414.0709229, -201.2179565, 440.7674866, -629.2322388, 615.2887573
2: -159.2985992, 457.8714600, -170.0539246, 486.5277710, -645.8262939, 627.9254150
3: -169.7159882, 574.7191162, -181.1163635, 610.3338013, -780.0498047, 755.8354492
4: -142.6433258, 530.3728638, -151.8613892, 562.1968994, -704.8402100, 682.2342529

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5662418, upper bound: 560.5804580
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -144.9611511, 442.4445801, -583.4120483, 575.3327026
1: -199.8453827, 436.4012756, -205.6305084, 448.5810547, -648.4264526, 642.0317993
2: -168.9626923, 481.6351929, -173.8231354, 495.1321106, -664.0947876, 655.4583130
3: -179.8964081, 603.5338135, -185.0542145, 620.5667114, -800.4631348, 788.5879517
4: -150.7288818, 555.7738037, -155.0763397, 571.7691040, -722.4979858, 710.8500977

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5785609, upper bound: 560.5867142
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762971, upper bound: 560.5762971
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -182.4133911, 564.2683105, -696.9434204, 590.2576904
1: -188.4647369, 414.0709229, -259.2867432, 570.8690186, -759.3337402, 673.3576050
2: -159.2985992, 457.8714600, -219.3215942, 631.0125122, -790.3110352, 677.1930542
3: -169.7159882, 574.7191162, -233.7263641, 792.1260376, -961.8420410, 808.4453735
4: -142.6433258, 530.3728638, -196.4786377, 730.4000854, -873.0433960, 726.8514404

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836150, upper bound: 560.5825179
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5836150, upper bound: 560.5870092
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -187.6401978, 577.4680176, -718.4354858, 618.0117798
1: -199.8453827, 436.4012756, -266.6047974, 584.7266235, -784.5720215, 703.0061035
2: -168.9626923, 481.6351929, -225.4896240, 646.4403076, -815.4028931, 707.1248169
3: -179.8964081, 603.5338135, -240.2658081, 810.8040161, -990.7003784, 843.7994995
4: -150.7288818, 555.7738037, -201.8733978, 748.0454712, -898.7743530, 757.6472168

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5830293
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5875312
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -143.7614288, 438.8794250, -619.6893921, 699.8626099
1: -256.9301758, 563.2346191, -203.9528351, 444.7846375, -701.7148438, 767.1874390
2: -217.2973633, 622.8032227, -172.3946381, 490.9169922, -708.2143555, 795.1977539
3: -231.5204010, 780.8549194, -183.5071411, 615.3638306, -846.8842163, 964.3619995
4: -194.5097809, 720.4786377, -153.7962036, 567.0385742, -761.5483398, 874.2748413

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -143.7614288, 438.8794250, -721.2144775, 1014.3010864
1: -400.9938660, 882.0807495, -203.9528351, 444.7846375, -845.7770386, 1084.0543213
2: -338.4806824, 975.1448364, -172.3946381, 490.9169922, -829.3975830, 1144.7683105
3: -361.4702759, 1219.9283447, -183.5071411, 615.3638306, -976.8341064, 1402.0679932
4: -303.4025574, 1124.5421143, -153.7962036, 567.0385742, -870.0560303, 1277.9506836

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -204.9295197, 625.3498535, -806.1598511, 761.0307007
1: -256.9301758, 563.2346191, -290.3470154, 633.9213257, -890.5545044, 853.5816650
2: -217.2973633, 622.8032227, -245.0624542, 700.8681030, -917.3948975, 867.8655396
3: -231.5204010, 780.8549194, -261.6196899, 875.7595825, -1107.2799072, 1042.4746094
4: -194.5097809, 720.4786377, -219.3437500, 808.1102295, -1002.6199951, 939.8223877

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5530093
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -204.9295197, 625.3498535, -906.7163696, 1074.7626953
1: -400.9938660, 882.0807495, -290.3470154, 633.9213257, -1032.8138428, 1169.3693848
2: -338.4806824, 975.1448364, -245.0624542, 700.8681030, -1037.0482178, 1216.5529785
3: -361.4702759, 1219.9283447, -261.6196899, 875.7595825, -1235.9990234, 1479.2838135
4: -303.4025574, 1124.5421143, -219.3437500, 808.1102295, -1110.0552979, 1342.5434570

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5530093
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
time: 0.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.76 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5662418, upper bound: 560.5804580
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5785609, upper bound: 560.5867142
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5762971, upper bound: 560.5762971
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5836150, upper bound: 560.5825179
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5836150, upper bound: 560.5870092
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5830293
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5875312
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5840291
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5530093
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5530093
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -560.5564143, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -131.3625336, 403.6568604, -136.1899109, 417.4776611, -548.8402100, 539.8467407
1: -186.5859070, 409.8310852, -193.1795959, 422.6810608, -609.2669678, 603.0105591
2: -157.7171783, 453.1820374, -163.2959137, 466.5332031, -624.2503052, 616.4778442
3: -168.0166931, 568.8250122, -173.8428497, 585.2835693, -753.3001709, 742.6677246
4: -141.2146301, 524.9486084, -145.7618561, 539.1492920, -680.3638916, 670.7104492

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5603336, upper bound: 560.5624201
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -131.2017822, 403.0686340, -172.8807831, 535.2350464, -666.4368286, 575.9494019
1: -186.3875732, 409.3093567, -245.4352875, 540.7169800, -727.1045532, 654.7446289
2: -157.5500793, 452.6371765, -207.1515350, 597.0952148, -754.6452026, 659.7885742
3: -167.8424835, 568.0755005, -221.2331238, 747.9669800, -915.8094482, 789.3085938
4: -141.0823822, 524.3125000, -185.4791870, 689.3464966, -830.4287720, 709.7916870

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5597462, upper bound: 560.5592721
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -139.5760651, 425.9514771, -139.2262726, 424.4381714, -564.0140991, 565.1777344
1: -197.8670349, 431.9136353, -197.4750214, 430.2053223, -628.0723877, 629.3886719
2: -167.2946320, 476.6651917, -166.9647980, 474.7520752, -642.0466919, 643.6300049
3: -178.1063843, 597.2978516, -177.6734161, 595.0906982, -773.1970215, 774.9711304
4: -149.2242126, 550.0437622, -148.8875122, 548.3344727, -697.5586548, 698.9311523

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5650988, upper bound: 560.5676168
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5685381, upper bound: 560.5703383
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -139.7596283, 426.4479065, -174.8837280, 540.7937622, -680.5534058, 601.3316650
1: -198.1336517, 432.4884033, -248.7527924, 546.7118530, -744.8455200, 681.2410278
2: -167.5226288, 477.3510132, -210.0197754, 603.6167603, -771.1394043, 687.3706055
3: -178.3519745, 598.0966187, -224.1028748, 755.8086548, -934.1606445, 822.1994629
4: -149.4413147, 550.8092651, -187.7642975, 696.3437500, -845.7849121, 738.5735474

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5647785, upper bound: 560.5654586
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5681921, upper bound: 560.5681921
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -182.5155792, 572.6014404, -705.2765503, 590.3598633
1: -188.4647369, 414.0709229, -259.6409302, 577.7786865, -766.2434082, 673.7118530
2: -159.2985992, 457.8714600, -219.5419617, 638.3512573, -797.6497803, 677.4134521
3: -169.7159882, 574.7191162, -234.1709137, 803.0667114, -972.7826538, 808.8898315
4: -142.6433258, 530.3728638, -196.8828735, 739.0787354, -881.7220459, 727.2557373

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792557, upper bound: 560.5565538
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5759393, upper bound: 560.5736040
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5742260, upper bound: 560.5736649
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -185.2178802, 569.7943726, -702.4694824, 593.0621948
1: -188.4647369, 414.0709229, -263.1491394, 577.0369263, -765.5016479, 677.2198486
2: -159.2985992, 457.8714600, -222.5697784, 637.9903564, -797.2889404, 680.4412231
3: -169.7159882, 574.7191162, -237.1614532, 800.1052856, -969.8212280, 811.8805542
4: -142.6433258, 530.3728638, -199.2773132, 738.2121582, -880.8554688, 729.6501465

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5792557, upper bound: 560.5569898
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5759393, upper bound: 560.5796876
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5742260, upper bound: 560.5797456
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -182.5155792, 572.6014404, -713.5689087, 612.8870850
1: -199.8453827, 436.4012756, -259.6409302, 577.7786865, -777.6240845, 696.0422363
2: -168.9626923, 481.6351929, -219.5419617, 638.3512573, -807.3138428, 701.1770630
3: -179.8964081, 603.5338135, -234.1709137, 803.0667114, -982.9630737, 837.7045898
4: -150.7288818, 555.7738037, -196.8828735, 739.0787354, -889.8076172, 752.6566772

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851063, upper bound: 560.5571043
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5822631
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5830293
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -185.2178802, 569.7943726, -710.7618408, 615.5894165
1: -199.8453827, 436.4012756, -263.1491394, 577.0369263, -776.8823242, 699.5502930
2: -168.9626923, 481.6351929, -222.5697784, 637.9903564, -806.9529419, 704.2049561
3: -179.8964081, 603.5338135, -237.1614532, 800.1052856, -980.0016479, 840.6952515
4: -150.7288818, 555.7738037, -199.2773132, 738.2121582, -888.9410400, 755.0511475

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5851063, upper bound: 560.5605218
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5867754
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5873176
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -142.0272980, 432.8376465, -613.6476440, 698.1284790
1: -256.9301758, 563.2346191, -201.5145264, 439.0659790, -695.9959717, 764.7491455
2: -217.2973633, 622.8032227, -170.3594208, 484.7265015, -702.0237427, 793.1626587
3: -231.5204010, 780.8549194, -181.3276215, 607.2824097, -838.8027344, 962.1825562
4: -194.5097809, 720.4786377, -151.9913788, 559.7343140, -754.2440796, 872.4700317

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830293, upper bound: 560.5871537
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5875144, upper bound: 560.5875144
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -180.8099976, 556.1011963, -736.9111938, 736.9111938
1: -256.9301758, 563.2346191, -256.9301758, 563.2346191, -820.1647949, 820.1647949
2: -217.2973633, 622.8032227, -217.2973633, 622.8032227, -840.1005249, 840.1004639
3: -231.5204010, 780.8549194, -231.5204010, 780.8549194, -1012.3753052, 1012.3753052
4: -194.5097809, 720.4786377, -194.5097809, 720.4786377, -914.9884033, 914.9884033

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830293, upper bound: 560.5871537
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5875144, upper bound: 560.5875144
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -142.0272980, 432.8376465, -715.1726685, 1012.5636597
1: -400.9938660, 882.0807495, -201.5145264, 439.0659790, -840.0598145, 1081.6159668
2: -338.4806824, 975.1448364, -170.3594208, 484.7265015, -823.2069702, 1142.7338867
3: -361.4702759, 1219.9283447, -181.3276215, 607.2824097, -968.7526245, 1399.8906250
4: -303.4025574, 1124.5421143, -151.9913788, 559.7343140, -862.7576294, 1276.1480713

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5571043, upper bound: 560.5834658
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5834658
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -180.8099976, 556.1011963, -838.4362793, 1050.6790771
1: -400.9938660, 882.0807495, -256.9301758, 563.2346191, -963.8558960, 1136.0776367
2: -338.4806824, 975.1448364, -217.2973633, 622.8032227, -961.1027832, 1188.7288818
3: -361.4702759, 1219.9283447, -231.5204010, 780.8549194, -1142.3251953, 1449.2137451
4: -303.4025574, 1124.5421143, -194.5097809, 720.4786377, -1023.5626221, 1317.7183838

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5571043, upper bound: 560.5834658
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5834658
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -174.4791870, 539.6496582, -192.1437073, 600.2207031, -774.0372925, 731.7933350
1: -248.1029053, 546.1126099, -272.9247742, 606.6093750, -853.9981079, 819.0373535
2: -209.8444672, 603.7604980, -230.2797089, 670.2797852, -878.9628296, 834.0401001
3: -223.6338501, 757.5394287, -246.2029266, 839.5767822, -1062.5511475, 1003.7423096
4: -187.9699097, 698.5946655, -206.6136169, 772.2291260, -959.9182129, 905.2082520

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5567092
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5571043
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -179.9389954, 553.3643799, -202.0720215, 616.1679077, -796.0129395, 755.4363403
1: -255.6901550, 560.4927368, -286.2727661, 624.7288208, -880.0571289, 846.7654419
2: -216.2466278, 619.7924194, -241.6374359, 690.7543945, -906.1545410, 861.4298706
3: -230.4080200, 777.0308228, -257.9499512, 862.9776001, -1093.3559570, 1034.9805908
4: -193.5745850, 716.9623413, -216.2983704, 796.3719482, -989.9190063, 933.2606812

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5601178
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5605162
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -275.2983398, 853.9934082, -192.1437073, 600.2207031, -873.8294067, 1043.6555176
1: -391.1903687, 863.2158203, -272.9247742, 606.6093750, -995.2907715, 1133.0909424
2: -330.1687012, 954.2076416, -230.2797089, 670.2797852, -997.7664185, 1180.7706299
3: -352.7252197, 1194.0845947, -246.2029266, 839.5767822, -1190.2239990, 1438.0064697
4: -296.0940552, 1100.1905518, -206.6136169, 772.2291260, -1066.4906006, 1305.4407959

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5530093
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5530093
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -281.7373962, 870.2901001, -202.0720215, 616.1679077, -896.7744141, 1069.8380127
1: -400.1309509, 880.1045532, -286.2727661, 624.7288208, -1022.6949463, 1163.2839355
2: -337.7579346, 972.9761963, -241.6374359, 690.7543945, -1026.1368408, 1210.9174805
3: -360.6914062, 1217.1898193, -257.9499512, 862.9776001, -1222.2177734, 1472.7762451
4: -302.7581482, 1122.0321045, -216.2983704, 796.3719482, -1097.5535889, 1336.9295654

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5564143
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5564143
time: 0.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.12 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5651492, upper bound: 560.5744788
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5650988, upper bound: 560.5676168
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5685381, upper bound: 560.5703383
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5647785, upper bound: 560.5654586
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5681921, upper bound: 560.5681921
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5759393, upper bound: 560.5736040
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5742260, upper bound: 560.5736649
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5759393, upper bound: 560.5796876
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5742260, upper bound: 560.5797456
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5822631
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5830293
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5867754
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5890584, upper bound: 560.5873176
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5830293, upper bound: 560.5871537
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5875144, upper bound: 560.5875144
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5830293, upper bound: 560.5871537
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5875144, upper bound: 560.5875144
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5571043, upper bound: 560.5834658
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5834658
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5571043, upper bound: 560.5834658
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5605162, upper bound: 560.5834658
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5567092
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5571043
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5601178
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5789846, upper bound: 560.5605162
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5530093
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5530093
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5564143
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -560.5530093, upper bound: 560.5564143

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -136.1899109, 417.4776611, -544.4749756, 525.8750610
1: -180.3264771, 395.6703491, -193.1795959, 422.6810608, -603.0075684, 588.8499756
2: -152.4571991, 437.5143433, -163.2959137, 466.5332031, -618.9904175, 600.8102417
3: -162.3547211, 549.1265259, -173.8428497, 585.2835693, -747.6382446, 722.9692993
4: -136.4553528, 506.8262329, -145.7618561, 539.1492920, -675.6046143, 652.5880737

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5627443, upper bound: 560.5704573
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5624997, upper bound: 560.5691700
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -136.1899109, 417.4776611, -579.5214233, 634.8546143
1: -229.8305664, 505.0910034, -193.1795959, 422.6810608, -652.5115967, 698.2706299
2: -194.1427002, 558.3701172, -163.2959137, 466.5332031, -660.6759033, 721.6660156
3: -207.1919708, 699.0578003, -173.8428497, 585.2835693, -792.4754639, 872.9005127
4: -174.0971069, 645.9729614, -145.7618561, 539.1492920, -713.2463989, 791.7348022

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5627443, upper bound: 560.5704573
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5624997, upper bound: 560.5691700
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -172.8807831, 535.2350464, -662.2322998, 562.5659180
1: -180.3264771, 395.6703491, -245.4352875, 540.7169800, -721.0434570, 641.1055908
2: -152.4571991, 437.5143433, -207.1515350, 597.0952148, -749.5523071, 644.6658936
3: -162.3547211, 549.1265259, -221.2331238, 747.9669800, -910.3217163, 770.3596191
4: -136.4553528, 506.8262329, -185.4791870, 689.3464966, -825.8018799, 692.3053589

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5633762
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5744788
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -172.8807831, 535.2350464, -697.2788086, 671.5454712
1: -229.8305664, 505.0910034, -245.4352875, 540.7169800, -770.5475464, 750.5263062
2: -194.1427002, 558.3701172, -207.1515350, 597.0952148, -791.2379150, 765.5216675
3: -207.1919708, 699.0578003, -221.2331238, 747.9669800, -955.1589355, 920.2908936
4: -174.0971069, 645.9729614, -185.4791870, 689.3464966, -863.4436035, 831.4521484

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5633762
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5744788
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -135.3642731, 413.0863647, -138.3004150, 421.5645142, -556.9287109, 551.3867798
1: -192.1270752, 418.8994751, -196.1954956, 427.2998352, -619.4268188, 615.0949707
2: -162.4387817, 462.2810059, -165.8808594, 471.5223694, -633.9609985, 628.1618042
3: -172.8903503, 578.9973145, -176.5112610, 591.0147705, -763.9050903, 755.5085449
4: -144.8634033, 533.2100830, -147.9146271, 544.5723877, -689.4357300, 681.1246948

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5633702
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5676168
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -143.9149780, 443.4729004, -138.0397797, 421.0758972, -564.9907837, 581.5125732
1: -204.0855408, 448.4763184, -195.7399139, 426.7401428, -630.8255615, 644.2162476
2: -172.4415131, 494.4348450, -165.4918518, 470.8983459, -643.3398438, 659.9266968
3: -183.6944275, 620.2651978, -176.1270142, 590.3549194, -774.0493164, 796.3921509
4: -153.8500824, 569.6215210, -147.6117249, 543.8135986, -697.6636353, 717.2332764

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5661297
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5703383
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -135.4431763, 413.2496338, -174.2558746, 538.8038330, -674.2470093, 587.5054932
1: -192.2435760, 419.1416321, -247.8562164, 544.7184448, -736.9619751, 666.9977417
2: -162.5388794, 462.5991211, -209.2637177, 601.4240112, -763.9628906, 671.8628540
3: -172.9993134, 579.3356323, -223.2977142, 753.0286255, -926.0279541, 802.6333008
4: -144.9650726, 533.5526123, -187.0900726, 693.7924194, -838.7574463, 720.6425781

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5592612
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5654586
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -143.7166290, 442.6920776, -172.6407318, 534.4219360, -678.1385498, 615.3328247
1: -203.8123932, 447.7777405, -245.7261200, 540.1389160, -743.9512329, 693.5037842
2: -172.2124176, 493.7200928, -207.4375916, 596.2566528, -768.4689941, 701.1577148
3: -183.4550476, 619.2537842, -221.2948914, 746.6773682, -930.1323853, 840.5487061
4: -153.6612091, 568.7390747, -185.3870544, 687.7036743, -841.3648682, 754.1260986

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5671606, upper bound: 560.5621008
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5671606, upper bound: 560.5681858
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -179.5771942, 563.5303955, -684.9261475, 553.8947144
1: -172.1784973, 379.8592529, -255.3724976, 568.6021729, -740.7806396, 635.2317505
2: -145.6561737, 420.1729736, -215.9608612, 628.2790527, -773.9352417, 636.1338501
3: -155.1810760, 527.8892212, -230.3332520, 790.4585571, -945.6396484, 758.2224731
4: -130.6282196, 486.8346863, -193.6914215, 727.4226685, -858.0509033, 680.5261230

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5648096, upper bound: 560.5665619
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5633991, upper bound: 560.5636809
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5645485, upper bound: 560.5091307
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5717810, upper bound: 560.5708439
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -125.5253906, 384.8227234, -179.6137238, 563.6346436, -689.1600342, 564.4364014
1: -178.2565460, 390.9218750, -255.4628143, 568.7098999, -746.9664307, 646.3845825
2: -150.7256012, 432.3974304, -216.0317688, 628.3438721, -779.0693970, 648.4290771
3: -160.4758301, 542.5620117, -230.4118347, 790.5711670, -951.0469971, 772.9738159
4: -134.9489594, 500.8383789, -193.7385406, 727.4996338, -862.4486084, 694.5769043

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5639632, upper bound: 560.5091859
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5711968, upper bound: 560.5709039
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -182.4328613, 561.5969849, -682.9927979, 556.7503662
1: -172.1784973, 379.8592529, -259.1158447, 568.6376343, -740.8161621, 638.9750977
2: -145.6561737, 420.1729736, -219.1809998, 628.7236938, -774.3798828, 639.3539429
3: -155.1810760, 527.8892212, -233.5310211, 788.5831299, -943.7642212, 761.4201660
4: -130.6282196, 486.8346863, -196.2522736, 727.4414673, -858.0697021, 683.0869751

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5650780, upper bound: 560.5681472
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5643782, upper bound: 560.5719464
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5761877, upper bound: 560.5774120
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5770454, upper bound: 560.5790698
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -125.5253906, 384.8227234, -182.1566010, 560.1222534, -685.6476440, 566.9792480
1: -178.2565460, 390.9218750, -258.7437134, 567.3281250, -745.5846558, 649.6655273
2: -150.7256012, 432.3974304, -218.8648529, 627.3150024, -778.0405884, 651.2620850
3: -160.4758301, 542.5620117, -233.1958466, 786.6949463, -947.1707764, 775.7578735
4: -134.9489594, 500.8383789, -195.9679413, 725.8654785, -860.8144531, 696.8062134

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5743804, upper bound: 560.5774700
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5739068, upper bound: 560.5791278
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.0930405, 373.0067749, -181.0134888, 567.9155884, -691.0086060, 554.0202637
1: -174.2050018, 379.3984070, -257.4959412, 573.0648193, -747.2697144, 636.8943481
2: -147.2826996, 419.4069824, -217.7435608, 633.1546021, -780.4373169, 637.1505127
3: -157.0052032, 524.2232666, -232.2327423, 796.4822388, -953.4874268, 756.4559937
4: -131.4749298, 483.7127380, -195.2718048, 732.9823608, -864.4572754, 678.9845581

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5725509, upper bound: 560.5729941
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5722596
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5822631
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.0707550, 418.1671753, -179.4096375, 563.2401123, -700.3108521, 597.5767822
1: -194.3686218, 424.1262207, -255.1684875, 568.3041382, -762.6727295, 679.2945557
2: -164.3190765, 468.1112061, -215.7700043, 627.8949585, -792.2140503, 683.8812256
3: -174.9659729, 586.4363403, -230.1895294, 789.8938599, -964.8597412, 816.6257935
4: -146.5706940, 540.0454712, -193.5126953, 727.0172119, -873.5878906, 733.5581665

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5729475
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5830293
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.0930405, 373.0067749, -183.6386414, 564.8054810, -687.8984985, 556.6453857
1: -174.2050018, 379.3984070, -260.8968811, 572.0551758, -746.2600708, 640.2952881
2: -147.2826996, 419.4069824, -220.6848145, 632.5191040, -779.8017578, 640.0917969
3: -157.0052032, 524.2232666, -235.1254578, 793.1663208, -950.1715088, 759.3486938
4: -131.4749298, 483.7127380, -197.5935974, 731.8302612, -863.3051758, 681.3063354

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5726131, upper bound: 560.5740493
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880737, upper bound: 560.5856481
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5867723
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -137.0707550, 418.1671753, -182.1187439, 560.4826660, -697.5532837, 600.2858887
1: -194.3686218, 424.1262207, -258.7048950, 567.6376953, -762.0062256, 682.8311157
2: -164.3190765, 468.1112061, -218.8044586, 627.6580200, -791.9771118, 686.9156494
3: -174.9659729, 586.4363403, -233.2074738, 787.0381470, -962.0039673, 819.6437988
4: -146.5706940, 540.0454712, -195.9231567, 726.2929688, -872.8636475, 735.9685059

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5880737, upper bound: 560.5858879
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5873176
time: 1.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -137.0606537, 419.9544067, -595.3063965, 687.5924683
1: -249.4921265, 555.4519653, -194.5000610, 425.5913391, -675.0833740, 749.9519653
2: -210.9501648, 613.7087402, -164.3952484, 469.8338013, -680.7838135, 778.1040039
3: -224.9954834, 771.8947754, -175.0689850, 589.1027832, -814.0982666, 946.9637451
4: -189.1553650, 710.2475586, -146.8185272, 542.7926636, -731.9479980, 857.0659790

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822631, upper bound: 560.5890584
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5830293, upper bound: 560.5890584
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -140.6679688, 428.7019043, -606.2393188, 686.4314575
1: -252.2587128, 552.8828125, -199.5493011, 434.8987122, -687.1574097, 752.4320679
2: -213.3510437, 611.4344482, -168.7020416, 480.1236572, -693.4746704, 780.1364746
3: -227.3259888, 766.4622803, -179.5744324, 601.4710693, -828.7970581, 946.0366821
4: -191.0043335, 707.2424316, -150.5088501, 554.2896118, -745.2939453, 857.7512817

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5839872
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5894410
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -174.4791870, 539.6496582, -715.0017700, 725.0109253
1: -249.4921265, 555.4519653, -248.1029053, 546.1126099, -795.6046143, 803.5548706
2: -210.9501648, 613.7087402, -209.8444672, 603.7604980, -814.7106323, 823.5531006
3: -224.9954834, 771.8947754, -223.6338501, 757.5394287, -982.5349121, 995.5286255
4: -189.1553650, 710.2475586, -187.9699097, 698.5946655, -887.7500000, 898.2173462

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825179, upper bound: 560.5826687
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5825179, upper bound: 560.5871537
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -179.9389954, 553.3643799, -730.9017944, 725.7025146
1: -252.2587128, 552.8828125, -255.6901550, 560.4927368, -812.7514648, 808.5729370
2: -213.3510437, 611.4344482, -216.2466278, 619.7924194, -833.1433105, 827.6810913
3: -227.3259888, 766.4622803, -230.4080200, 777.0308228, -1004.3568115, 996.8701782
4: -191.0043335, 707.2424316, -193.5745850, 716.9623413, -907.9666138, 900.8170166

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5830293
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5875144
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -137.0606537, 419.9544067, -695.9662476, 1002.3802490
1: -392.7666626, 875.2170410, -194.5000610, 425.5913391, -818.2734375, 1067.1754150
2: -331.3969727, 966.5505371, -164.3952484, 469.8338013, -801.2307129, 1127.7647705
3: -354.2966614, 1211.7148438, -175.0689850, 589.1027832, -943.3992920, 1384.2584229
4: -297.2891541, 1113.6004639, -146.8185272, 542.7926636, -839.6615601, 1259.4992676

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5482536, upper bound: 560.5847447
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5482536, upper bound: 560.5851260
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -140.6679688, 428.7019043, -708.7003784, 1003.3308716
1: -397.6463318, 874.5075073, -199.5493011, 434.8987122, -832.5450439, 1071.9635010
2: -335.6636353, 966.8534546, -168.7020416, 480.1236572, -815.7872925, 1132.6773682
3: -358.4500122, 1209.4127197, -179.5744324, 601.4710693, -959.9210815, 1387.3059082
4: -300.8917236, 1114.8928223, -150.5088501, 554.2896118, -854.7585449, 1264.8391113

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5569898, upper bound: 560.5792557
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5569898, upper bound: 560.5851063
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -174.4791870, 539.6496582, -815.6615601, 1039.1422119
1: -392.7666626, 875.2170410, -248.1029053, 546.1126099, -938.4993286, 1119.8531494
2: -331.3969727, 966.5505371, -209.8444672, 603.7604980, -934.9059448, 1172.2923584
3: -354.2966614, 1211.7148438, -223.6338501, 757.5394287, -1111.8360596, 1431.9832764
4: -297.2891541, 1113.6004639, -187.9699097, 698.5946655, -995.5807495, 1299.7205811

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5567092, upper bound: 560.5789846
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5567092, upper bound: 560.5834658
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -179.9389954, 553.3643799, -833.3628540, 1041.9351807
1: -397.6463318, 874.5075073, -255.6901550, 560.4927368, -957.7385864, 1127.1506348
2: -335.6636353, 966.8534546, -216.2466278, 619.7924194, -955.2456055, 1179.2764893
3: -358.4500122, 1209.4127197, -230.4080200, 777.0308228, -1135.4808350, 1437.2701416
4: -300.8917236, 1114.8928223, -193.5745850, 716.9623413, -1017.4831543, 1306.9571533

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601178, upper bound: 560.5789846
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5601178, upper bound: 560.5834658
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -192.1437073, 600.2207031, -774.9829102, 742.6755371
1: -249.4921265, 555.4519653, -272.9247742, 606.6093750, -855.4420166, 828.3767090
2: -210.9501648, 613.7087402, -230.2797089, 670.2797852, -880.1010132, 843.9883423
3: -224.9954834, 771.8947754, -246.2029266, 839.5767822, -1063.9682617, 1018.0976562
4: -189.1553650, 710.2475586, -206.6136169, 772.2291260, -961.1251831, 916.8612061

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5754769, upper bound: 560.5366606
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5766283, upper bound: 560.5541157
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -192.1437073, 600.2207031, -776.8962402, 737.9072266
1: -252.2587128, 552.8828125, -272.9247742, 606.6093750, -857.9746094, 825.8075562
2: -213.3510437, 611.4344482, -230.2797089, 670.2797852, -882.3200073, 841.7140503
3: -227.3259888, 766.4622803, -246.2029266, 839.5767822, -1066.1044922, 1012.6651001
4: -191.0043335, 707.2424316, -206.6136169, 772.2291260, -962.8558960, 913.8560791

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5754769, upper bound: 560.5370082
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5766283, upper bound: 560.5544380
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -202.0720215, 616.1679077, -791.3304443, 752.6038208
1: -249.4921265, 555.4519653, -286.2727661, 624.7288208, -873.7642822, 841.7246704
2: -210.9501648, 613.7087402, -241.6374359, 690.7543945, -900.5983887, 855.3461914
3: -224.9954834, 771.8947754, -257.9499512, 862.9776001, -1087.9069824, 1029.8444824
4: -189.1553650, 710.2475586, -216.2983704, 796.3719482, -985.4510498, 926.5458984

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722342, upper bound: 560.5541448
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729240, upper bound: 560.5489295
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -202.0720215, 616.1679077, -793.6105347, 747.8355103
1: -252.2587128, 552.8828125, -286.2727661, 624.7288208, -876.6071167, 839.1555176
2: -213.3510437, 611.4344482, -241.6374359, 690.7543945, -903.2404175, 853.0718994
3: -227.3259888, 766.4622803, -257.9499512, 862.9776001, -1090.2677002, 1024.4121094
4: -191.0043335, 707.2424316, -216.2983704, 796.3719482, -987.3458862, 923.5407715

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5722342, upper bound: 560.5544336
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729240, upper bound: 560.5493285
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -202.0720215, 616.1679077, -891.0036621, 1066.4307861
1: -392.7666626, 875.2170410, -286.2727661, 624.7288208, -1015.2995605, 1157.6280518
2: -331.3969727, 966.5505371, -241.6374359, 690.7543945, -1019.5722656, 1203.9173584
3: -354.2966614, 1211.7148438, -257.9499512, 862.9776001, -1215.8146973, 1466.0526123
4: -297.2891541, 1113.6004639, -216.2983704, 796.3719482, -1092.0521240, 1327.8710938

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5432456, upper bound: 560.5499299
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5439775, upper bound: 560.5446380
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -202.0720215, 616.1679077, -895.0279541, 1064.0288086
1: -397.6463318, 874.5075073, -286.2727661, 624.7288208, -1020.1926270, 1157.6080322
2: -335.6636353, 966.8534546, -241.6374359, 690.7543945, -1024.0253906, 1204.7259521
3: -358.4500122, 1209.4127197, -257.9499512, 862.9776001, -1219.9600830, 1464.7802734
4: -300.8917236, 1114.8928223, -216.2983704, 796.3719482, -1095.6711426, 1329.6668701

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5432456, upper bound: 560.5485396
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5439775, upper bound: 560.5445399
time: 0.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.22 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5627443, upper bound: 560.5704573
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5624997, upper bound: 560.5691700
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5627443, upper bound: 560.5704573
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5624997, upper bound: 560.5691700
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5633762
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5744788
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5633762
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5633762, upper bound: 560.5744788
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5633702
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5676168
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5661297
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5703383
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5592612
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5654586
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5671606, upper bound: 560.5621008
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5671606, upper bound: 560.5681858
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5645485, upper bound: 560.5091307
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5717810, upper bound: 560.5708439
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5639632, upper bound: 560.5091859
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5711968, upper bound: 560.5709039
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5761877, upper bound: 560.5774120
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5770454, upper bound: 560.5790698
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5743804, upper bound: 560.5774700
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5739068, upper bound: 560.5791278
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5722596
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5822631
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5729475
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5867243, upper bound: 560.5830293
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5880737, upper bound: 560.5856481
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5867723
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5880737, upper bound: 560.5858879
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5894410, upper bound: 560.5873176
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5822631, upper bound: 560.5890584
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5830293, upper bound: 560.5890584
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5839872
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5894410
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5825179, upper bound: 560.5826687
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5825179, upper bound: 560.5871537
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5830293
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5870092, upper bound: 560.5875144
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5482536, upper bound: 560.5847447
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5482536, upper bound: 560.5851260
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5569898, upper bound: 560.5792557
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5569898, upper bound: 560.5851063
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5567092, upper bound: 560.5789846
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5567092, upper bound: 560.5834658
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5601178, upper bound: 560.5789846
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5601178, upper bound: 560.5834658
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5754769, upper bound: 560.5366606
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5766283, upper bound: 560.5541157
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5754769, upper bound: 560.5370082
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5766283, upper bound: 560.5544380
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5722342, upper bound: 560.5541448
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5729240, upper bound: 560.5489295
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5722342, upper bound: 560.5544336
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5729240, upper bound: 560.5493285
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5432456, upper bound: 560.5499299
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5439775, upper bound: 560.5446380
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5432456, upper bound: 560.5485396
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 8.22
Output dim: 0, lower bound: -560.5439775, upper bound: 560.5445399

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -126.1525726, 387.0752869, -132.1052399, 404.9750977, -531.1276855, 519.1804810
1: -179.1739807, 393.0294495, -187.5813293, 410.0372314, -589.2111206, 580.6106567
2: -151.4801025, 434.5843201, -158.5652313, 452.6054382, -604.0854492, 593.1495361
3: -161.3045502, 545.4193115, -168.7631836, 567.5081787, -728.8127441, 714.1824341
4: -135.5757294, 503.4166260, -141.5232544, 522.7760010, -658.3516235, 644.9397583

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5665509, upper bound: 560.5704988
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5665509, upper bound: 560.5704988
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -125.8434982, 386.3254700, -140.8737488, 436.1983643, -562.0418701, 527.1991577
1: -178.6044922, 392.2078247, -199.9031830, 440.4911194, -619.0955811, 592.1110229
2: -150.9959564, 433.6847229, -168.8745728, 485.7982178, -636.7941284, 602.5592041
3: -160.8434448, 544.3952026, -179.8903503, 610.0370483, -770.8804932, 724.2855225
4: -135.2173309, 502.3297729, -150.7960663, 560.3543091, -695.5716553, 653.1257935

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5533986, upper bound: 560.5568849
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5573987, upper bound: 560.5577164
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -161.5199738, 497.0260315, -132.1052399, 404.9750977, -566.4950562, 629.1312866
1: -229.0888214, 503.4500732, -187.5813293, 410.0372314, -639.1259155, 691.0313721
2: -193.5176086, 556.5615234, -158.5652313, 452.6054382, -646.1228638, 715.1267700
3: -206.5252991, 696.7677002, -168.7631836, 567.5081787, -774.0334473, 865.5308228
4: -173.5415344, 643.8663940, -141.5232544, 522.7760010, -696.3175049, 785.3896484

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5528917, upper bound: 560.5581539
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5573955, upper bound: 560.5537975
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5546610, upper bound: 560.5596190
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5619828, upper bound: 560.5689750
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5603598, upper bound: 560.5685504
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -159.7617035, 492.2193604, -140.8737488, 436.1983643, -595.9600220, 633.0930176
1: -226.7593231, 498.4460449, -199.9031830, 440.4911194, -667.2504272, 698.3492432
2: -191.5233307, 550.9428101, -168.8745728, 485.7982178, -677.3215332, 719.8173218
3: -204.3439484, 689.8098145, -179.8903503, 610.0370483, -814.3809814, 869.7001343
4: -171.6825409, 637.2322388, -150.7960663, 560.3543091, -732.0368652, 788.0282593

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5526279, upper bound: 560.5564034
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5616889, upper bound: 560.5674952
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5600296, upper bound: 560.5668501
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -162.0438080, 498.6647949, -625.6619873, 551.7289429
1: -180.3264771, 395.6703491, -229.8305664, 505.0910034, -685.4174805, 625.5008545
2: -152.4571991, 437.5143433, -194.1427002, 558.3701172, -710.8273315, 631.6570435
3: -162.3547211, 549.1265259, -207.1919708, 699.0578003, -861.4125366, 756.3184814
4: -136.4553528, 506.8262329, -174.0971069, 645.9729614, -782.4283447, 680.9232178

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5661634, upper bound: 560.5625572
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649741, upper bound: 560.5623326
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -126.9973297, 389.6851501, -170.2176056, 527.2655640, -654.2628174, 559.9027710
1: -180.3264771, 395.6703491, -242.2979736, 533.0390015, -713.3654785, 637.9683228
2: -152.4571991, 437.5143433, -204.6186676, 588.2682495, -740.7254639, 642.1329956
3: -162.3547211, 549.1265259, -218.2718201, 736.6583252, -899.0130615, 767.3983154
4: -136.4553528, 506.8262329, -182.8941498, 678.2100220, -814.6654053, 689.7202759

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5661634, upper bound: 560.5695022
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649741, upper bound: 560.5683741
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -162.0438080, 498.6647949, -660.7084961, 660.7084961
1: -229.8305664, 505.0910034, -229.8305664, 505.0910034, -734.9215698, 734.9215698
2: -194.1427002, 558.3701172, -194.1427002, 558.3701172, -752.5128174, 752.5128174
3: -207.1919708, 699.0578003, -207.1919708, 699.0578003, -906.2497559, 906.2497559
4: -174.0971069, 645.9729614, -174.0971069, 645.9729614, -820.0700684, 820.0700073

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626012, upper bound: 560.5616620
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5610637, upper bound: 560.5610637
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -162.0438080, 498.6647949, -170.2176056, 527.2655640, -689.3093872, 668.8823242
1: -229.8305664, 505.0910034, -242.2979736, 533.0390015, -762.8695679, 747.3889771
2: -194.1427002, 558.3701172, -204.6186676, 588.2682495, -782.4109497, 762.9887695
3: -207.1919708, 699.0578003, -218.2718201, 736.6583252, -943.8502808, 917.3295898
4: -174.0971069, 645.9729614, -182.8941498, 678.2100220, -852.3070679, 828.8670654

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5626012, upper bound: 560.5728503
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5610637, upper bound: 560.5722484
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -135.3642731, 413.0863647, -126.1525726, 387.0752869, -522.4395142, 539.2389526
1: -192.1270752, 418.8994751, -179.1739807, 393.0294495, -585.1563110, 598.0734253
2: -162.4387817, 462.2810059, -151.4801025, 434.5843201, -597.0230713, 613.7609863
3: -172.8903503, 578.9973145, -161.3045502, 545.4193115, -718.3096313, 740.3018799
4: -144.8634033, 533.2100830, -135.5757294, 503.4166260, -648.2798462, 668.7857666

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564725, upper bound: 560.5538190
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5633702
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5633702
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -135.3642731, 413.0863647, -134.1187744, 408.7895508, -544.1537476, 547.2051392
1: -192.1270752, 418.8994751, -190.1599731, 414.4517212, -606.5786133, 609.0594482
2: -162.4387817, 462.2810059, -160.7957764, 457.2814331, -619.7202148, 623.0765991
3: -172.8903503, 578.9973145, -171.1233521, 573.0101318, -745.9004517, 750.1206665
4: -144.8634033, 533.2100830, -143.3551178, 527.6905518, -672.5538940, 676.5651855

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5564725, upper bound: 560.5548834
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5676168
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5649935, upper bound: 560.5676168
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -143.9149780, 443.4729004, -125.8434982, 386.3254700, -530.2402954, 569.3163452
1: -204.0855408, 448.4763184, -178.6044922, 392.2078247, -596.2933350, 627.0808105
2: -172.4415131, 494.4348450, -150.9959564, 433.6847229, -606.1262207, 645.4307861
3: -183.6944275, 620.2651978, -160.8434448, 544.3952026, -728.0895996, 781.1086426
4: -153.8500824, 569.6215210, -135.2173309, 502.3297729, -656.1796875, 704.8388672

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5589787, upper bound: 560.5567105
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5661297
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5661297
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -143.9149780, 443.4729004, -133.8572235, 408.1812439, -552.0961914, 577.3300171
1: -204.0855408, 448.4763184, -189.6599731, 413.7769165, -617.8624268, 638.1362915
2: -172.4415131, 494.4348450, -160.3712158, 456.5404968, -628.9819946, 654.8060303
3: -183.6944275, 620.2651978, -170.7186737, 572.1842041, -755.8786011, 790.9838867
4: -153.8500824, 569.6215210, -143.0411072, 526.8038330, -680.6538086, 712.6625977

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5589787, upper bound: 560.5576714
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5703383
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5683741, upper bound: 560.5703383
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -135.4431763, 413.2496338, -161.5199738, 497.0260315, -632.4692383, 574.7695923
1: -192.2435760, 419.1416321, -229.0888214, 503.4500732, -695.6936646, 648.2302856
2: -162.5388794, 462.5991211, -193.5176086, 556.5615234, -719.1004028, 656.1166992
3: -172.9993134, 579.3356323, -206.5252991, 696.7677002, -869.7669067, 785.8609619
4: -144.9650726, 533.5526123, -173.5415344, 643.8663940, -788.8314209, 707.0941162

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5592612
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5592612
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -135.4431763, 413.2496338, -169.7125244, 525.6881714, -661.1312866, 582.9621582
1: -192.2435760, 419.1416321, -241.5799561, 531.4573364, -723.7008667, 660.7214966
2: -162.5388794, 462.5991211, -204.0133209, 586.5290527, -749.0679321, 666.6124268
3: -172.9993134, 579.3356323, -217.6270447, 734.4514160, -907.4507446, 796.9626465
4: -144.9650726, 533.5526123, -182.3562012, 676.1798096, -821.1448364, 715.9088135

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5654586
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5637485, upper bound: 560.5654586
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -143.7166290, 442.6920776, -159.7617035, 492.2193604, -635.9359131, 602.4537964
1: -203.8123932, 447.7777405, -226.7593231, 498.4460449, -702.2584229, 674.5370483
2: -172.2124176, 493.7200928, -191.5233307, 550.9428101, -723.1552124, 685.2434082
3: -183.4550476, 619.2537842, -204.3439484, 689.8098145, -873.2648315, 823.5977173
4: -153.6612091, 568.7390747, -171.6825409, 637.2322388, -790.8934326, 740.4216309

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5671606, upper bound: 560.5621008
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5671606, upper bound: 560.5621008
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -143.7166290, 442.6920776, -168.0458832, 521.1298218, -664.8464355, 610.7379761
1: -203.8123932, 447.7777405, -239.3723450, 526.7046509, -730.5170288, 687.1500244
2: -172.2124176, 493.7200928, -202.1228027, 581.1757812, -753.3881226, 695.8428955
3: -183.4550476, 619.2537842, -215.5570221, 727.8520508, -911.3070679, 834.8107910
4: -153.6612091, 568.7390747, -180.5947876, 669.8779297, -823.5391235, 749.3337402

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5657754, upper bound: 560.5678578
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5670157, upper bound: 560.5680505
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -118.7336044, 367.1805420, -178.5022583, 564.0137939, -682.7473145, 545.6828003
1: -168.2996216, 372.4582214, -253.6605225, 568.0889282, -736.3885498, 626.1186523
2: -142.3676147, 412.0611267, -214.5731049, 627.7779541, -770.1455078, 626.6340942
3: -151.7606201, 517.9057007, -228.9562073, 791.0894165, -942.8500366, 746.8619385
4: -127.8246231, 477.5214233, -192.7851105, 727.4763184, -855.3008423, 670.3065186

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5645485, upper bound: 560.5091307
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=618.8850708007812
rel_dist={0: [-560.5900507657058, 560.5900507657057]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5891644, upper bound: 560.5873210
time: 1.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 0, lower bound: -560.5891644, upper bound: 560.5873210
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -149.3368835, 456.1983643, -602.7736816, 596.6884155
1: -207.9424896, 453.5178833, -211.8698273, 462.3305969, -670.2730713, 665.3876343
2: -175.7731476, 500.5854492, -179.0831451, 510.2579956, -686.0309448, 679.6685181
3: -187.1182709, 627.4591675, -190.6544800, 639.7701416, -826.8884277, 818.1136475
4: -156.8196411, 578.2070312, -159.7823792, 589.4827881, -746.3023682, 737.9893799

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 1.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
time: 0.92 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -188.6081696, 580.4994507, -147.8033752, 452.2457581, -640.8538208, 728.3028564
1: -267.9796143, 587.7565918, -209.7061768, 458.0008240, -725.9804688, 797.4627686
2: -226.6574402, 649.7827759, -177.2468872, 505.4262390, -732.0836182, 827.0296631
3: -241.4969330, 815.0511475, -188.6853638, 633.9469604, -875.4439087, 1003.7365112
4: -202.9143524, 751.9616089, -158.1368256, 583.9628906, -786.8771973, 910.0984497

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5582863, upper bound: 560.5833294
time: 0.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -560.5873168, upper bound: 560.5873168
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -560.5582863, upper bound: 560.5833294
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -146.5755005, 447.3516541, -593.9271240, 593.9270630
1: -207.9424896, 453.5178833, -207.9424896, 453.5178833, -661.4603882, 661.4603882
2: -175.7731476, 500.5854492, -175.7731476, 500.5854492, -676.3585815, 676.3585815
3: -187.1182709, 627.4591675, -187.1182709, 627.4591675, -814.5774536, 814.5774536
4: -156.8196411, 578.2070312, -156.8196411, 578.2070312, -735.0266724, 735.0266724

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5829077, upper bound: 560.5866327
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887013, upper bound: 560.5866820
time: 1.12 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -146.5755005, 447.3516541, -188.6081696, 580.4994507, -727.0748901, 635.9598389
1: -207.9424896, 453.5178833, -267.9796143, 587.7565918, -795.6990967, 721.4974976
2: -175.7731476, 500.5854492, -226.6574402, 649.7827759, -825.5559082, 727.2428589
3: -187.1182709, 627.4591675, -241.4969330, 815.0511475, -1002.1694336, 868.9561157
4: -156.8196411, 578.2070312, -202.9143524, 751.9616089, -908.7811890, 781.1213989

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5829077, upper bound: 560.5866327
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887013, upper bound: 560.5866820
time: 0.99 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -184.7147675, 568.3251953, -141.1664429, 431.0726318, -615.7874146, 709.4916382
1: -262.4612732, 575.5171509, -200.2674561, 436.7703857, -699.2316895, 775.7845459
2: -221.9783630, 636.3206177, -169.2718048, 482.0840759, -704.0623779, 805.5922852
3: -236.5144348, 797.9879761, -180.1711426, 604.2847900, -840.7991943, 978.1591187
4: -198.7134552, 736.2526855, -150.9880981, 556.7943115, -755.5077515, 887.2407227

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 1.15 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -182.0769043, 559.3459473, -203.3727570, 620.7866821, -802.8635254, 762.7186890
1: -258.6050110, 566.4321899, -288.1311035, 629.2644653, -887.6073608, 854.5632324
2: -218.7689667, 626.2164307, -243.1780701, 695.7202759, -913.7562256, 869.3945312
3: -233.0495605, 785.2961426, -259.6271667, 869.3163452, -1102.3659668, 1044.9233398
4: -195.8558502, 724.6314697, -217.6612854, 802.1063232, -997.9621582, 942.2927246

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5829077, upper bound: 560.5866327
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5887013, upper bound: 560.5866820
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5829077, upper bound: 560.5866327
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5887013, upper bound: 560.5866820
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -138.8587799, 427.7122498, -560.3874512, 546.7030640
1: -188.4647369, 414.0709229, -197.0638580, 432.9197388, -621.3844604, 611.1347046
2: -159.2985992, 457.8714600, -166.5251312, 477.8962097, -637.1948242, 624.3966064
3: -169.7159882, 574.7191162, -177.4154816, 599.8184814, -769.5344238, 752.1345215
4: -142.6433258, 530.3728638, -148.8072968, 552.3766479, -695.0199585, 679.1801758

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5823641, upper bound: 560.5885491
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5829077, upper bound: 560.5885491
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -142.8662567, 436.0070190, -576.9744263, 573.2378540
1: -199.8453827, 436.4012756, -202.6036682, 442.1155701, -641.9609375, 639.0049438
2: -168.9626923, 481.6351929, -171.2709351, 487.9936829, -656.9562988, 652.9060669
3: -179.8964081, 603.5338135, -182.3530731, 611.5355225, -791.4318848, 785.8869019
4: -150.7288818, 555.7738037, -152.7986450, 563.3402100, -714.0690308, 708.5724487

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5767787, upper bound: 560.5836858
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5755125, upper bound: 560.5755125
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -178.4145355, 553.5732422, -686.2482910, 586.2589111
1: -188.4647369, 414.0709229, -253.5682831, 559.7207031, -748.1854248, 667.6390381
2: -159.2985992, 457.8714600, -214.4987488, 618.6171265, -777.9157104, 672.3702393
3: -169.7159882, 574.7191162, -228.6124268, 776.9541626, -946.6701050, 803.3314819
4: -142.6433258, 530.3728638, -192.2437439, 716.1217041, -858.7650146, 722.6165771

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828506, upper bound: 560.5821992
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5828506, upper bound: 560.5866327
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -186.5339355, 573.9719238, -714.9393921, 616.9055176
1: -199.8453827, 436.4012756, -265.0297852, 581.2220459, -781.0674438, 701.4309692
2: -168.9626923, 481.6351929, -224.1533813, 642.5870361, -811.5495605, 705.7885742
3: -179.8964081, 603.5338135, -238.8502655, 805.9248047, -985.8211670, 842.3839722
4: -150.7288818, 555.7738037, -200.6824036, 743.5552979, -894.2841187, 756.4561768

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5822817
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5866820
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -141.1664429, 431.0726318, -611.8826294, 697.2676392
1: -256.9301758, 563.2346191, -200.2674561, 436.7703857, -693.7004395, 763.5020142
2: -217.2973633, 622.8032227, -169.2718048, 482.0840759, -699.3813477, 792.0748901
3: -231.5204010, 780.8549194, -180.1711426, 604.2847900, -835.8051758, 961.0260620
4: -194.5097809, 720.4786377, -150.9880981, 556.7943115, -751.3040771, 871.4667358

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -141.1664429, 431.0726318, -713.4077148, 1011.7030640
1: -400.9938660, 882.0807495, -200.2674561, 436.7703857, -837.7575073, 1080.3620605
2: -338.4806824, 975.1448364, -169.2718048, 482.0840759, -820.5645142, 1141.6391602
3: -361.4702759, 1219.9283447, -180.1711426, 604.2847900, -965.7550659, 1398.7257080
4: -303.4025574, 1124.5421143, -150.9880981, 556.7943115, -859.8147583, 1275.1367188

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -175.4611969, 550.0968018, -191.1413727, 588.6946411, -764.1558228, 741.2381592
1: -249.5059662, 555.1149902, -270.9805603, 596.1895142, -845.3684082, 826.0954590
2: -211.0275574, 613.2831421, -228.6572418, 658.9832764, -869.1707764, 841.9403687
3: -225.0511017, 771.3353882, -244.3275452, 824.0303955, -1049.0815430, 1015.6629639
4: -189.2517090, 709.8526001, -204.8976135, 759.4782104, -948.7299194, 914.7501831

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5512871, upper bound: 560.5530081
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -178.6566772, 548.5673828, -201.6406708, 615.1738892, -793.8233643, 750.2080078
1: -253.7421722, 555.6290283, -285.6575928, 623.6520386, -877.0686646, 841.2864380
2: -214.6605835, 614.3318481, -241.0963287, 689.5498047, -903.4069214, 855.4281616
3: -228.6759186, 770.2423706, -257.4002686, 861.5112915, -1090.1872559, 1027.6425781
4: -192.1897583, 710.7742920, -215.8126831, 794.9490967, -987.1388550, 926.5869751

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5545906, upper bound: 560.5530081
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.74 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5823641, upper bound: 560.5885491
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5829077, upper bound: 560.5885491
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5767787, upper bound: 560.5836858
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5755125, upper bound: 560.5755125
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5828506, upper bound: 560.5821992
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5828506, upper bound: 560.5866327
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5822817
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5866820
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5833294
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5512871, upper bound: 560.5530081
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5545906, upper bound: 560.5530081
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.74
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -127.4752502, 391.8775635, -121.2539825, 371.4299927, -498.9052429, 513.1315308
1: -180.9078217, 398.0218201, -171.8547516, 377.0232544, -557.9310303, 569.8765259
2: -152.8687592, 440.2983398, -145.1641083, 416.7271423, -569.5958862, 585.4624023
3: -163.0142365, 552.6443481, -154.8881989, 521.9956665, -685.0098877, 707.5325317
4: -137.0061493, 510.2400513, -129.8431549, 481.6774292, -618.6835938, 640.0831909

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5757170, upper bound: 560.5785503
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5738409, upper bound: 560.5786358
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -129.8933105, 399.0732422, -135.1776428, 416.1254883, -546.0187378, 534.2508545
1: -184.5666351, 405.2795410, -191.8800354, 421.2743225, -605.8409424, 597.1595459
2: -155.9988251, 448.1823120, -162.1302185, 465.0618286, -621.0606079, 610.3125000
3: -166.2044525, 562.4584351, -172.7523804, 583.5761108, -749.7805786, 735.2108154
4: -139.6956787, 519.0984497, -144.8735504, 537.4656372, -677.1612549, 663.9719849

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5757170, upper bound: 560.5801567
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5745189, upper bound: 560.5802544
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -137.6683655, 419.9030762, -137.0802460, 417.7843018, -555.4525146, 556.9832764
1: -195.1598206, 425.7623596, -194.3770294, 423.5309448, -618.6907959, 620.1394043
2: -165.0133057, 469.8437805, -164.3520508, 467.3854675, -632.3988037, 634.1958008
3: -175.6546478, 588.7524414, -174.9114990, 585.7525635, -761.4072266, 763.6638184
4: -147.1612091, 542.1858521, -146.5555115, 539.6383057, -686.7993774, 688.7413330

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5630685, upper bound: 560.5662748
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5675384, upper bound: 560.5693621
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -139.1332245, 424.4353943, -172.6885529, 534.2579346, -673.3911743, 597.1239624
1: -197.2481232, 430.4775391, -245.6992950, 540.1271973, -737.3753052, 676.1766968
2: -166.7762146, 475.1492310, -207.4611664, 596.2542114, -763.0303955, 682.6101685
3: -177.5538635, 595.3077393, -221.3419800, 746.5944214, -924.1481934, 816.6495972
4: -148.7737885, 548.2614136, -185.4606934, 687.6614380, -836.4351196, 733.7220459

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5630507, upper bound: 560.5640933
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5672861, upper bound: 560.5672861
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -182.5155792, 572.6014404, -705.2765503, 590.3598633
1: -188.4647369, 414.0709229, -259.6409302, 577.7786865, -766.2434082, 673.7118530
2: -159.2985992, 457.8714600, -219.5419617, 638.3512573, -797.6497803, 677.4134521
3: -169.7159882, 574.7191162, -234.1709137, 803.0667114, -972.7826538, 808.8898315
4: -142.6433258, 530.3728638, -196.8828735, 739.0787354, -881.7220459, 727.2557373

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5758032, upper bound: 560.5733964
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5739081, upper bound: 560.5734598
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -132.6752014, 407.8443604, -185.2178802, 569.7943726, -702.4694824, 593.0621948
1: -188.4647369, 414.0709229, -263.1491394, 577.0369263, -765.5016479, 677.2198486
2: -159.2985992, 457.8714600, -222.5697784, 637.9903564, -797.2889404, 680.4412231
3: -169.7159882, 574.7191162, -237.1614532, 800.1052856, -969.8212280, 811.8805542
4: -142.6433258, 530.3728638, -199.2773132, 738.2121582, -880.8554688, 729.6501465

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5758032, upper bound: 560.5794708
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5739081, upper bound: 560.5795282
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -182.5155792, 572.6014404, -713.5689087, 612.8870850
1: -199.8453827, 436.4012756, -259.6409302, 577.7786865, -777.6240845, 696.0422363
2: -168.9626923, 481.6351929, -219.5419617, 638.3512573, -807.3138428, 701.1770630
3: -179.8964081, 603.5338135, -234.1709137, 803.0667114, -982.9630737, 837.7045898
4: -150.7288818, 555.7738037, -196.8828735, 739.0787354, -889.8076172, 752.6566772

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822727, upper bound: 560.5553025
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5816924
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5822817
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -140.9674835, 430.3715820, -185.2178802, 569.7943726, -710.7618408, 615.5894165
1: -199.8453827, 436.4012756, -263.1491394, 577.0369263, -776.8823242, 699.5502930
2: -168.9626923, 481.6351929, -222.5697784, 637.9903564, -806.9529419, 704.2049561
3: -179.8964081, 603.5338135, -237.1614532, 800.1052856, -980.0016479, 840.6952515
4: -150.7288818, 555.7738037, -199.2773132, 738.2121582, -888.9410400, 755.0511475

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822727, upper bound: 560.5574377
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5833797
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5862666
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -142.0272980, 432.8376465, -613.6476440, 698.1284790
1: -256.9301758, 563.2346191, -201.5145264, 439.0659790, -695.9959717, 764.7491455
2: -217.2973633, 622.8032227, -170.3594208, 484.7265015, -702.0237427, 793.1626587
3: -231.5204010, 780.8549194, -181.3276215, 607.2824097, -838.8027344, 962.1825562
4: -194.5097809, 720.4786377, -151.9913788, 559.7343140, -754.2440796, 872.4700317

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822817, upper bound: 560.5866327
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866820, upper bound: 560.5866820
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -180.8099976, 556.1011963, -180.8099976, 556.1011963, -736.9111938, 736.9111938
1: -256.9301758, 563.2346191, -256.9301758, 563.2346191, -820.1647949, 820.1647949
2: -217.2973633, 622.8032227, -217.2973633, 622.8032227, -840.1005249, 840.1004639
3: -231.5204010, 780.8549194, -231.5204010, 780.8549194, -1012.3753052, 1012.3753052
4: -194.5097809, 720.4786377, -194.5097809, 720.4786377, -914.9884033, 914.9884033

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822817, upper bound: 560.5866327
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866820, upper bound: 560.5866820
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -142.0272980, 432.8376465, -715.1726685, 1012.5636597
1: -400.9938660, 882.0807495, -201.5145264, 439.0659790, -840.0598145, 1081.6159668
2: -338.4806824, 975.1448364, -170.3594208, 484.7265015, -823.2069702, 1142.7338867
3: -361.4702759, 1219.9283447, -181.3276215, 607.2824097, -968.7526245, 1399.8906250
4: -303.4025574, 1124.5421143, -151.9913788, 559.7343140, -862.7576294, 1276.1480713

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553025, upper bound: 560.5827044
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5822723
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -282.3350525, 872.2695312, -180.8099976, 556.1011963, -838.4362793, 1050.6790771
1: -400.9938660, 882.0807495, -256.9301758, 563.2346191, -963.8558960, 1136.0776367
2: -338.4806824, 975.1448364, -217.2973633, 622.8032227, -961.1027832, 1188.7288818
3: -361.4702759, 1219.9283447, -231.5204010, 780.8549194, -1142.3251953, 1449.2137451
4: -303.4025574, 1124.5421143, -194.5097809, 720.4786377, -1023.5626221, 1317.7183838

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5553025, upper bound: 560.5827044
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5822723
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -173.9818268, 545.3361206, -186.7869110, 575.8393555, -749.8211670, 732.1230469
1: -247.3425751, 550.4191284, -264.9095459, 583.1602783, -829.8249512, 815.3284912
2: -209.2527161, 608.1182251, -223.5692596, 644.5829468, -852.6853638, 831.6875000
3: -223.1071014, 764.8545532, -238.8392181, 806.1994019, -1029.3062744, 1003.6937866
4: -187.6557922, 703.8392334, -200.3348083, 742.8424683, -930.3347168, 904.1740723

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -175.2769165, 539.1247559, -195.0157623, 596.2408447, -771.5177002, 734.1405029
1: -249.1506348, 546.0527344, -276.5230713, 604.5554199, -852.9663086, 822.5758057
2: -210.8144073, 603.6874390, -233.4237213, 668.4369507, -878.0498047, 837.1111450
3: -224.5493469, 757.0430298, -249.1905670, 835.1870117, -1059.5421143, 1006.2335205
4: -188.7283478, 698.3693848, -208.9422302, 770.2210693, -958.6340942, 907.3115845

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.42 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5757170, upper bound: 560.5785503
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5738409, upper bound: 560.5786358
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5757170, upper bound: 560.5801567
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5745189, upper bound: 560.5802544
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5630685, upper bound: 560.5662748
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5675384, upper bound: 560.5693621
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5630507, upper bound: 560.5640933
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5672861, upper bound: 560.5672861
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5758032, upper bound: 560.5733964
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5739081, upper bound: 560.5734598
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5758032, upper bound: 560.5794708
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5739081, upper bound: 560.5795282
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5816924
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5822817
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5833797
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5885549, upper bound: 560.5862666
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5822817, upper bound: 560.5866327
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5866820, upper bound: 560.5866820
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5822817, upper bound: 560.5866327
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5866820, upper bound: 560.5866820
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5553025, upper bound: 560.5827044
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5822723
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5553025, upper bound: 560.5827044
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5574377, upper bound: 560.5822723
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5523808, upper bound: 560.5557014
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -560.5557014, upper bound: 560.5557014

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -115.1692123, 355.4615784, -113.9614105, 350.0037231, -465.1729431, 469.4229736
1: -163.2759857, 360.8652649, -161.3385468, 355.1200867, -518.3960571, 522.2037964
2: -138.0662842, 399.3146667, -136.3381348, 392.5302124, -530.5963135, 535.6527710
3: -147.2551575, 501.6575317, -145.4467010, 491.9097595, -639.1649170, 647.1042480
4: -123.9035721, 462.8061523, -121.9816360, 453.6090393, -577.5125732, 584.7877808

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5737216, upper bound: 560.5785503
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5737216, upper bound: 560.5785503
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -119.8726730, 367.4564514, -117.9502869, 360.4530029, -480.3256836, 485.4067383
1: -170.0737000, 373.4407043, -167.1163483, 366.1175842, -536.1911011, 540.5570679
2: -143.7655792, 413.2218628, -141.1792755, 404.7624817, -548.5278931, 554.4011230
3: -153.1945953, 518.5075073, -150.6090088, 506.8626099, -660.0571899, 669.1165161
4: -128.7870026, 478.9249878, -126.2750015, 467.8725586, -596.6594849, 605.1998901

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5738409, upper bound: 560.5786358
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5737216, upper bound: 560.5786358
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5737216, upper bound: 560.5786358
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -119.9395523, 369.5744324, -129.8497314, 400.9416809, -520.8811035, 499.4241638
1: -170.1360779, 375.1086121, -184.1229858, 405.6511536, -575.7872314, 559.2315674
2: -143.9309845, 414.9451294, -155.6208801, 447.8398743, -591.7708130, 570.5659790
3: -153.3290863, 521.2783813, -165.8623047, 562.3681030, -715.6972046, 687.1406860
4: -129.0749664, 480.7872009, -139.1634827, 517.5419922, -646.6169434, 619.9506836

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5753625, upper bound: 560.5734001
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5753625, upper bound: 560.5801567
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -122.9624710, 376.7843933, -130.3300018, 400.1216125, -523.0840454, 507.1143799
1: -174.6874847, 382.8531494, -184.9228821, 405.3723755, -580.0598145, 567.7760010
2: -147.7069855, 423.4980469, -156.2801971, 447.6143188, -595.3212891, 579.7781982
3: -157.2525024, 531.3054199, -166.4589539, 561.4011841, -718.6536865, 697.7644043
4: -132.2426300, 490.4811707, -139.6428986, 517.2351074, -649.4777222, 630.1240845

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5735020, upper bound: 560.5735019
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5735019, upper bound: 560.5802544
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -133.4728851, 407.0810242, -134.8050995, 410.7968140, -544.2697144, 541.8861084
1: -189.4360657, 412.7913513, -191.2546387, 416.4573364, -605.8934326, 604.0459595
2: -160.1713562, 455.5184937, -161.7084198, 459.5449219, -619.7161865, 617.2268677
3: -170.4549866, 570.5092163, -172.0707245, 575.8255005, -746.2804565, 742.5797729
4: -142.8160706, 525.4063721, -144.1831055, 530.4909668, -673.3070068, 669.5894775

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5516900, upper bound: 560.5596054
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5630462, upper bound: 560.5662748
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5630462, upper bound: 560.5662748
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -142.1548767, 437.9074707, -135.2945862, 412.5809631, -554.7358398, 573.2020264
1: -201.5781250, 442.8255310, -191.7102966, 418.1776123, -619.7555542, 634.5357666
2: -170.3343353, 488.1871948, -162.0781555, 461.4537659, -631.7880859, 650.2653809
3: -181.4281006, 612.4308472, -172.5728912, 578.4324951, -759.8605347, 785.0037231
4: -151.9546509, 562.4199829, -144.6191711, 532.6879272, -684.6425781, 707.0390625

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5460174, upper bound: 560.5582498
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5674251, upper bound: 560.5654048
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5674251, upper bound: 560.5693621
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -134.7306824, 410.9417419, -171.0135040, 528.9533081, -663.6839600, 581.9552612
1: -191.2357330, 416.8384399, -243.3094177, 534.8135986, -726.0493164, 660.1477051
2: -161.6885681, 460.0787964, -205.4458618, 590.4099121, -752.0984497, 665.5245972
3: -172.0900574, 576.1394653, -219.1952209, 739.1849365, -911.2748413, 795.3347168
4: -144.2046204, 530.6345215, -183.6658173, 680.8586426, -825.0631104, 714.3002319

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5354011, upper bound: 560.5564995
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5332770, upper bound: 560.5353739
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -142.8309174, 439.8109741, -169.0900116, 524.0603638, -666.8912964, 608.9010010
1: -202.5638580, 444.9037476, -240.8493805, 529.6000977, -732.1639404, 685.7529907
2: -171.1578064, 490.5738220, -203.3264008, 584.4614258, -755.6192017, 693.9002075
3: -182.3283997, 615.2477417, -216.8385315, 731.9700317, -914.2984009, 832.0863037
4: -152.7198944, 565.0902100, -181.6502075, 673.8198242, -826.5396118, 746.7403564

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5443009, upper bound: 560.5650960
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -560.5421881, upper bound: 560.5421881
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -177.3329315, 556.5875854, -677.9833984, 551.6504517
1: -172.1784973, 379.8592529, -252.1039429, 561.5847168, -733.7631836, 631.9631348
2: -145.6561737, 420.1729736, -213.2253113, 620.5748291, -766.2310181, 633.3981934
3: -155.1810760, 527.8892212, -227.3954163, 780.7974854, -935.9785767, 755.2845459
4: -130.6282196, 486.8346863, -191.2510681, 718.4887695, -849.1170044, 678.0856323

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5568992, upper bound: 560.5619972
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5588340, upper bound: 560.5081262
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5715012, upper bound: 560.5706329
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -125.5253906, 384.8227234, -177.2802734, 556.3965454, -681.9219360, 562.1029663
1: -178.2565460, 390.9218750, -252.0921326, 561.3965454, -739.6530151, 643.0139771
2: -150.7256012, 432.3974304, -213.1971741, 620.2785645, -771.0041504, 645.5944824
3: -160.4758301, 542.5620117, -227.3841705, 780.4900513, -940.9658813, 769.9461060
4: -134.9489594, 500.8383789, -191.2029572, 718.1591797, -853.1081543, 692.0412598

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5728485, upper bound: 560.5723085
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5734257, upper bound: 560.5729398
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -121.3957825, 374.3175354, -180.4087067, 555.6730957, -677.0688477, 554.7262573
1: -172.1784973, 379.8592529, -256.1772461, 562.5600586, -734.7385254, 636.0364990
2: -145.6561737, 420.1729736, -216.7156372, 622.0222168, -767.6784058, 636.8885498
3: -155.1810760, 527.8892212, -230.8888245, 780.2668457, -935.4479370, 758.7778931
4: -130.6282196, 486.8346863, -194.0537720, 719.6400757, -850.2683105, 680.8884277

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5563708, upper bound: 560.5628549
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5729885, upper bound: 560.5757936
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5762772, upper bound: 560.5788520
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -125.5253906, 384.8227234, -179.6940002, 552.3219604, -677.8473511, 564.5166016
1: -178.2565460, 390.9218750, -255.1933594, 559.5081177, -737.7646484, 646.1151733
2: -150.7256012, 432.3974304, -215.8820496, 618.7188721, -769.4444580, 648.2794189
3: -160.4758301, 542.5620117, -230.0013885, 775.8931274, -936.3689575, 772.5634155
4: -134.9489594, 500.8383789, -193.3042297, 715.9200439, -850.8690186, 694.1425781

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5736736, upper bound: 560.5769805
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5744678, upper bound: 560.5789071
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.0930405, 373.0067749, -179.8927155, 564.4250488, -687.5180664, 552.8994751
1: -174.2050018, 379.3984070, -255.8927155, 569.5644531, -743.7692871, 635.2911377
2: -147.2826996, 419.4069824, -216.4058838, 629.2968140, -776.5794678, 635.8128662
3: -157.0052032, 524.2232666, -230.7823181, 791.5816040, -948.5867920, 755.0055542
4: -131.4749298, 483.7127380, -194.0684509, 728.4420166, -859.9169312, 677.7811890

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5837273, upper bound: 560.5712923
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5837273, upper bound: 560.5816924
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.0707550, 418.1671753, -177.2249908, 556.3374023, -693.4081421, 595.3921509
1: -194.3686218, 424.1262207, -252.0263367, 561.3394165, -755.7080078, 676.1523438
2: -164.3190765, 468.1112061, -213.1377869, 620.1707764, -784.4898682, 681.2490234
3: -174.9659729, 586.4363403, -227.3718567, 780.1698608, -955.1357422, 813.8080444
4: -146.5706940, 540.0454712, -191.1534119, 718.1325073, -864.7031860, 731.1988525

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5842002, upper bound: 560.5718509
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5842002, upper bound: 560.5822817
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -123.0930405, 373.0067749, -182.5052032, 561.1926880, -684.2857056, 555.5119629
1: -174.2050018, 379.3984070, -259.2869568, 568.4648438, -742.6697388, 638.6853638
2: -147.2826996, 419.4069824, -219.3421631, 628.5758057, -775.8584595, 638.7491455
3: -157.0052032, 524.2232666, -233.6680603, 788.1571655, -945.1623535, 757.8913574
4: -131.4749298, 483.7127380, -196.3875122, 727.2224121, -858.6973267, 680.1002197

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866837, upper bound: 560.5788303
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866837, upper bound: 560.5813953
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -137.0707550, 418.1671753, -179.9800110, 553.8147583, -690.8853760, 598.1472168
1: -194.3686218, 424.1262207, -255.6456299, 560.9406738, -755.3092041, 679.7717285
2: -164.3190765, 468.1112061, -216.2418671, 620.2451782, -784.5642700, 684.3530884
3: -174.9659729, 586.4363403, -230.4700317, 777.7171021, -952.6829224, 816.9063721
4: -146.5706940, 540.0454712, -193.6278229, 717.7770386, -864.3477173, 733.6732788

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5862578, upper bound: 560.5846791
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5887013, upper bound: 560.5862666
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -133.9785309, 412.0298462, -587.3818359, 684.5103149
1: -249.4921265, 555.4519653, -190.1550293, 417.3027954, -666.7949219, 745.6069946
2: -210.9501648, 613.7087402, -160.7037659, 460.7222290, -671.6722412, 774.4124146
3: -224.9954834, 771.8947754, -171.1971741, 577.9628906, -802.9583740, 943.0918579
4: -189.1553650, 710.2475586, -143.6183929, 532.3942871, -721.5496826, 853.8658447

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5816924, upper bound: 560.5885549
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5822817, upper bound: 560.5885549
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -138.8908844, 423.2550659, -600.7924805, 684.6543579
1: -252.2587128, 552.8828125, -196.9775696, 429.4224243, -681.6811523, 749.8603516
2: -213.3510437, 611.4344482, -166.5347137, 474.0766602, -687.4277344, 777.9691162
3: -227.3259888, 766.4622803, -177.2803345, 593.8216553, -821.1476440, 943.7425537
4: -191.0043335, 707.2424316, -148.5739746, 547.1324463, -738.1367798, 855.8164062

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5833797, upper bound: 560.5887013
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866820, upper bound: 560.5887013
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -175.3521118, 550.5317993, -170.5222168, 528.9976807, -704.3497925, 721.0540161
1: -249.4921265, 555.4519653, -242.4266357, 535.0822144, -784.5743408, 797.8786011
2: -210.9501648, 613.7087402, -205.0552979, 591.4942017, -802.4443359, 818.7640381
3: -224.9954834, 771.8947754, -218.5586700, 742.5067749, -967.5022583, 990.4534302
4: -189.1553650, 710.2475586, -183.7691193, 684.4771729, -873.6325684, 894.0165405

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821992, upper bound: 560.5822070
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5821992, upper bound: 560.5866327
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -177.5374146, 545.7634888, -178.9233704, 550.1491699, -727.6865845, 724.6868896
1: -252.2587128, 552.8828125, -254.2405548, 557.2713013, -809.5300293, 807.1232910
2: -213.3510437, 611.4344482, -215.0187378, 616.2555542, -829.6064453, 826.4531860
3: -227.3259888, 766.4622803, -229.1051788, 772.5504761, -999.8764648, 995.5674438
4: -191.0043335, 707.2424316, -192.4812012, 712.8442993, -903.8485718, 899.7236328

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866327, upper bound: 560.5822817
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5866327, upper bound: 560.5866820
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -133.9785309, 412.0298462, -688.0417480, 999.3087769
1: -392.7666626, 875.2170410, -190.1550293, 417.3027954, -809.9512329, 1062.8297119
2: -331.3969727, 966.5505371, -160.7037659, 460.7222290, -792.1191406, 1124.0727539
3: -354.2966614, 1211.7148438, -171.1971741, 577.9628906, -932.2593994, 1380.3853760
4: -297.2891541, 1113.6004639, -143.6183929, 532.3942871, -829.2653809, 1256.2957764

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5257707, upper bound: 560.5686935
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5462281, upper bound: 560.5758330
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -138.8908844, 423.2550659, -703.2535400, 1001.5527344
1: -397.6463318, 874.5075073, -196.9775696, 429.4224243, -827.0687256, 1069.3795166
2: -335.6636353, 966.8534546, -166.5347137, 474.0766602, -809.7402954, 1130.5040283
3: -358.4500122, 1209.4127197, -177.2803345, 593.8216553, -952.2716675, 1385.0076904
4: -300.8917236, 1114.8928223, -148.5739746, 547.1324463, -847.5781860, 1262.9012451

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5448390, upper bound: 560.5707795
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5472836, upper bound: 560.5762870
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -276.0119324, 868.1258545, -170.5222168, 528.9976807, -805.0096436, 1035.1881104
1: -392.7666626, 875.2170410, -242.4266357, 535.0822144, -927.4770508, 1114.1846924
2: -331.3969727, 966.5505371, -205.0552979, 591.4942017, -922.6615601, 1167.5076904
3: -354.2966614, 1211.7148438, -218.5586700, 742.5067749, -1096.8034668, 1426.9150391
4: -297.2891541, 1113.6004639, -183.7691193, 684.4771729, -981.4974365, 1295.5245361

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5426481, upper bound: 560.5749687
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -560.5539558, upper bound: 560.5816613
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -279.9984741, 864.6758423, -178.9233704, 550.1491699, -830.1476440, 1040.9190674
1: -397.6463318, 874.5075073, -254.2405548, 557.2713013, -954.5098267, 1125.6927490
2: -335.6636353, 966.8534546, -215.0187378, 616.2555542, -951.6991577, 1178.0400391
3: -358.4500122, 1209.4127197, -229.1051788, 772.5504761, -1131.0003662, 1435.9644775
4: -300.8917236, 1114.8928223, -192.4812012, 712.8442993, -1013.3325195, 1305.8621826

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=618.8850708007812
rel_dist={0: [-560.5891721066854, 560.5891721066855]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1120.38 seconds
