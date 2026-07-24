## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 743.673742927666


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447)
1: (-226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553)
2: (-160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006)
3: (-390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531)
4: (-263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.70 + 1.85 = 2.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -743.6886167, upper bound: 743.6886167

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817214, upper bound: 743.6819280
time: 0.52 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -743.6817214, upper bound: 743.6819280
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 0, lower bound: -743.6802824, upper bound: 743.6802824

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -128.5790863, 670.3025513, -134.0998993, 698.8281250, -827.4071655, 804.4024048
1: -210.4309387, 796.4016113, -219.3339996, 830.1954346, -1040.6262207, 1015.7355957
2: -148.6245575, 824.4069214, -155.0493164, 859.5734253, -1008.1979370, 979.4562378
3: -362.4870300, 699.0101929, -377.8995972, 728.8269043, -1091.3137207, 1076.9097900
4: -244.7495728, 707.2666626, -255.3451080, 737.4572144, -982.2067871, 962.6117554

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6803546, upper bound: 743.6792488
time: 0.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6806099, upper bound: 743.6803410
time: 0.60 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -250.1162415, 1337.2148438, -132.0244751, 690.9362793, -941.0524902, 1469.2392578
1: -412.2479858, 1588.0107422, -215.8887939, 820.3369141, -1232.5849609, 1803.8995361
2: -290.3746338, 1641.2213135, -152.8052368, 849.8156738, -1140.1901855, 1794.0262451
3: -711.9147949, 1393.7313232, -372.3732300, 719.3096313, -1431.2243652, 1766.1044922
4: -478.8499146, 1407.6657715, -251.5644379, 728.5810547, -1207.4309082, 1659.2302246

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791732, upper bound: 743.6783583
time: 0.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794644, upper bound: 743.6794644
time: 0.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.99 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -743.6803546, upper bound: 743.6792488
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -743.6806099, upper bound: 743.6803410
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -743.6791732, upper bound: 743.6783583
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.99
Output dim: 0, lower bound: -743.6794644, upper bound: 743.6794644

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -120.0708618, 626.9495850, -117.8189087, 615.6799927, -735.7508545, 744.7684937
1: -196.5260773, 744.6201172, -192.7268829, 730.8889160, -927.4149780, 937.3469849
2: -138.8246002, 771.2864380, -136.2743683, 757.7134399, -896.5380249, 907.5607910
3: -338.6886292, 652.9367676, -332.3695374, 640.5405273, -979.2290039, 985.3061523
4: -228.6001129, 661.0349731, -224.3954010, 648.8972168, -877.4972534, 885.4303589

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6642988, upper bound: 743.6636261
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796505, upper bound: 743.6780467
time: 0.70 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -112.8097687, 589.5702515, -140.9153900, 737.6492920, -850.4589844, 730.4856567
1: -184.2080231, 700.0789795, -229.1755371, 875.7017212, -1059.9097900, 929.2545166
2: -130.3366699, 725.2761230, -162.9476624, 907.6109619, -1037.9475098, 888.2237549
3: -317.5670166, 613.4398804, -396.5287170, 767.2593384, -1084.8264160, 1009.9686279
4: -214.5980377, 621.1304321, -269.0603943, 776.5700073, -991.1679688, 890.1907349

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6642988, upper bound: 743.6639883
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796837, upper bound: 743.6782806
time: 0.59 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -241.6555939, 1293.4482422, -115.1913910, 604.3644409, -846.0200195, 1408.6396484
1: -398.3694458, 1535.7913818, -188.3076630, 716.9603271, -1115.3295898, 1724.0989990
2: -280.5727234, 1587.5083008, -133.3654175, 743.7992554, -1024.3718262, 1720.8737793
3: -688.1551514, 1347.3814697, -325.1696167, 627.6037598, -1315.7589111, 1672.5510254
4: -462.7062988, 1361.2049561, -219.5247345, 636.6154175, -1099.3217773, 1580.7293701

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6704810, upper bound: 743.6681049
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6562805, upper bound: 743.6592065
time: 0.64 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -234.7287750, 1256.8201904, -136.5489349, 717.7695923, -952.4983521, 1393.3691406
1: -386.4918823, 1492.1151123, -222.1980591, 851.5682373, -1238.0599365, 1714.3132324
2: -272.5178528, 1542.5157471, -158.1401062, 883.1951294, -1155.7128906, 1700.6558838
3: -667.9785156, 1308.9216309, -384.6334229, 745.3094482, -1413.2879639, 1693.5550537
4: -449.4403076, 1322.4339600, -261.0271301, 755.2985229, -1204.7387695, 1583.4610596

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785645, upper bound: 743.6788378
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785626, upper bound: 743.6785626
time: 0.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.94 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6642988, upper bound: 743.6636261
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6796505, upper bound: 743.6780467
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6642988, upper bound: 743.6639883
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6796837, upper bound: 743.6782806
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6704810, upper bound: 743.6681049
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6562805, upper bound: 743.6592065
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6785645, upper bound: 743.6788378
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 0, lower bound: -743.6785626, upper bound: 743.6785626

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -118.6061707, 618.9829102, -115.8709335, 605.0268555, -723.6329956, 734.8538208
1: -193.7272949, 735.1195068, -189.4338379, 718.4623413, -912.1896362, 924.5533447
2: -136.9372101, 761.2526245, -133.9353333, 744.1986694, -881.1358643, 895.1879883
3: -333.8558044, 644.9784546, -326.6411133, 630.1162720, -963.9719238, 971.6193848
4: -225.4477539, 653.0494995, -220.5769501, 638.0830078, -863.5307617, 873.6264038

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796505, upper bound: 743.6780467
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796505, upper bound: 743.6780467
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -108.2893906, 567.2600708, -137.2250214, 717.0651855, -825.3544922, 704.4849243
1: -176.6360931, 673.2980957, -223.1006927, 851.5553589, -1028.1914062, 896.3988037
2: -125.0020599, 697.5504150, -158.6093750, 882.2067261, -1007.2088013, 856.1597900
3: -304.4656677, 589.7185669, -385.9307251, 746.7243652, -1051.1899414, 975.6492920
4: -205.7451630, 597.3646851, -261.9083557, 755.4885864, -961.2337646, 859.2730713

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796837, upper bound: 743.6782806
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796837, upper bound: 743.6782806
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -226.9039764, 1214.6738281, -133.8510590, 703.5563965, -930.4602661, 1348.5246582
1: -373.6329956, 1441.8728027, -217.7160034, 834.5863037, -1208.2189941, 1659.5888672
2: -263.3541565, 1491.1245117, -154.9778442, 865.7882690, -1129.1424561, 1646.1022949
3: -645.6913452, 1264.2871094, -376.9244080, 730.1977539, -1375.8889160, 1641.2114258
4: -434.1915588, 1277.9824219, -255.7750397, 740.2227783, -1174.4141846, 1533.7574463

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785645, upper bound: 743.6788361
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6766243, upper bound: 743.6768568
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -230.9691467, 1238.3603516, -133.4669342, 701.6425781, -932.6116943, 1371.8271484
1: -380.3873596, 1469.9202881, -217.2132111, 832.3626709, -1212.7500000, 1687.1333008
2: -268.1502380, 1519.7677002, -154.5618744, 863.4622192, -1131.6124268, 1674.3292236
3: -657.3591309, 1288.7581787, -375.9608765, 728.2247314, -1385.5836182, 1664.7189941
4: -442.1538391, 1302.3151855, -255.0532684, 738.2435913, -1180.3970947, 1557.3684082

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6689780, upper bound: 743.6679365
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6659367, upper bound: 743.6659367
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.99 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6796505, upper bound: 743.6780467
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6796505, upper bound: 743.6780467
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6796837, upper bound: 743.6782806
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6796837, upper bound: 743.6782806
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6766243, upper bound: 743.6768568
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6689780, upper bound: 743.6679365
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 1.99
Output dim: 0, lower bound: -743.6659367, upper bound: 743.6659367

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -118.6061707, 618.9829102, -111.1582870, 580.7720947, -699.3781738, 730.1411743
1: -193.7272949, 735.1195068, -181.8727875, 689.7561646, -883.4834595, 916.9923096
2: -136.9372101, 761.2526245, -128.4882812, 714.3176880, -851.2548828, 889.7409058
3: -333.8558044, 644.9784546, -313.5140991, 604.7847290, -938.6405029, 958.4924927
4: -225.4477539, 653.0494995, -211.6058197, 612.3679810, -837.8156738, 864.6553345

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -118.6061707, 618.9829102, -231.4289246, 1240.7276611, -1359.3338623, 850.4117432
1: -193.7272949, 735.1195068, -381.6576843, 1473.0316162, -1666.7589111, 1116.7772217
2: -136.9372101, 761.2526245, -268.7124329, 1522.6791992, -1659.6164551, 1029.9650879
3: -333.8558044, 644.9784546, -659.4547729, 1292.0083008, -1625.8636475, 1304.4332275
4: -225.4477539, 653.0494995, -443.1968689, 1305.5261230, -1530.9737549, 1096.2462158

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -108.2893906, 567.2600708, -132.5191650, 692.6243286, -800.9136353, 699.7791138
1: -176.6360931, 673.2980957, -215.5250854, 822.6453857, -999.2814331, 888.8231812
2: -125.0020599, 697.5504150, -153.1771088, 852.1047974, -977.1068726, 850.7274780
3: -304.4656677, 589.7185669, -372.8997192, 721.2971191, -1025.7628174, 962.6182861
4: -205.7451630, 597.3646851, -252.9785004, 729.6998291, -935.4450073, 850.3431396

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6782806
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6781575
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -108.2893906, 567.2600708, -245.0944214, 1306.9317627, -1415.2210693, 812.3544312
1: -176.6360931, 673.2980957, -402.4073792, 1551.9871826, -1728.6232910, 1075.7054443
2: -125.0020599, 697.5504150, -284.3610229, 1604.2645264, -1729.2666016, 981.9114380
3: -304.4656677, 589.7185669, -696.2764282, 1362.7675781, -1667.2332764, 1285.9949951
4: -205.7451630, 597.3646851, -469.2458496, 1376.7092285, -1582.4542236, 1066.6105957

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6782806
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6781575
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -225.0367126, 1204.9183350, -125.4384689, 659.6718140, -884.7084961, 1330.3565674
1: -370.6385193, 1430.2484131, -203.9035187, 782.3321533, -1152.9707031, 1634.1519775
2: -261.1991272, 1479.1784668, -145.1545258, 811.8204956, -1073.0195312, 1624.3327637
3: -640.4933472, 1253.9710693, -353.1893005, 684.0723267, -1324.5655518, 1607.1604004
4: -430.6247559, 1267.6408691, -239.5682526, 693.6756592, -1124.3004150, 1507.2091064

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6766114, upper bound: 743.6777270
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -224.1718597, 1200.4833984, -129.2769165, 679.4330444, -903.6048584, 1329.7600098
1: -369.1628723, 1424.9338379, -210.2142334, 805.8634644, -1175.0263672, 1635.1479492
2: -260.1855164, 1473.7222900, -149.6743011, 836.2318726, -1096.4173584, 1623.3964844
3: -637.9282227, 1249.2148438, -363.9529114, 704.7927246, -1342.7207031, 1613.1677246
4: -428.9396667, 1262.8742676, -247.0024109, 714.7019043, -1143.6412354, 1509.8767090

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761861, upper bound: 743.6768483
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6766243, upper bound: 743.6768568
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.82 seconds
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6794239, upper bound: 743.6780467
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6782806
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6781575
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6782806
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6793942, upper bound: 743.6781575
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6766114, upper bound: 743.6777270
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6761861, upper bound: 743.6768483
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -743.6766243, upper bound: 743.6768568

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -112.5535049, 587.8195190, -111.1582870, 580.7720947, -693.3255615, 698.9777832
1: -183.7729950, 697.9183960, -181.8727875, 689.7561646, -873.5291748, 879.7911987
2: -129.9520111, 722.9826050, -128.4882812, 714.3176880, -844.2697144, 851.4708862
3: -316.7823486, 611.9615479, -313.5140991, 604.7847290, -921.5670776, 925.4756470
4: -213.9270020, 619.8825684, -211.6058197, 612.3679810, -826.2948608, 831.4884033

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830842, upper bound: 743.6821182
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820360, upper bound: 743.6819924
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -126.2742004, 661.5014038, -111.1582870, 580.7720947, -707.0462646, 772.6595459
1: -205.1608276, 785.3093872, -181.8727875, 689.7561646, -894.9169312, 967.1821899
2: -145.9014435, 813.7514648, -128.4882812, 714.3176880, -860.2191162, 942.2397461
3: -354.9310303, 687.7292480, -313.5140991, 604.7847290, -959.7157593, 1001.2433472
4: -240.8751373, 696.0109253, -211.6058197, 612.3679810, -853.2431030, 907.6167603

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830842, upper bound: 743.6821284
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820360, upper bound: 743.6819924
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -112.5535049, 587.8195190, -231.3919830, 1240.5435791, -1353.0970459, 819.2114868
1: -183.7729950, 697.9183960, -381.5971069, 1472.8135986, -1656.5865479, 1079.5152588
2: -129.9520111, 722.9826050, -268.6696777, 1522.4512939, -1652.4033203, 991.6522827
3: -316.7823486, 611.9615479, -659.3499756, 1291.8146973, -1608.5969238, 1271.3115234
4: -213.9270020, 619.8825684, -443.1259766, 1305.3284912, -1519.2554932, 1063.0085449

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6608289
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6537467, upper bound: 743.6526691
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -126.2742004, 661.5014038, -231.3481293, 1240.3249512, -1366.5991211, 892.8494263
1: -205.1608276, 785.3093872, -381.5251465, 1472.5546875, -1677.7155762, 1166.8344727
2: -145.9014435, 813.7514648, -268.6188965, 1522.1807861, -1668.0821533, 1082.3701172
3: -354.9310303, 687.7292480, -659.2258301, 1291.5849609, -1646.5159912, 1346.9550781
4: -240.8751373, 696.0109253, -443.0419312, 1305.0942383, -1545.8522949, 1139.0528564

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6608289
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6537467, upper bound: 743.6526691
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -112.5535049, 587.8195190, -132.5191650, 692.6243286, -805.1777954, 720.3386841
1: -183.7729950, 697.9183960, -215.5250854, 822.6453857, -1006.4183960, 913.4434204
2: -129.9520111, 722.9826050, -153.1771088, 852.1047974, -982.0568237, 876.1596680
3: -316.7823486, 611.9615479, -372.8997192, 721.2971191, -1038.0794678, 984.8612671
4: -213.9270020, 619.8825684, -252.9785004, 729.6998291, -943.6267700, 872.8610840

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818314, upper bound: 743.6819892
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817848, upper bound: 743.6818247
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -126.3022079, 661.6420898, -132.5191650, 692.6243286, -818.9264526, 794.1611938
1: -205.2046204, 785.4786377, -215.5250854, 822.6453857, -1027.8498535, 1001.0037231
2: -145.9328613, 813.9240112, -153.1771088, 852.1047974, -998.0376587, 967.1011353
3: -355.0084839, 687.8810425, -372.8997192, 721.2971191, -1076.3055420, 1060.7800293
4: -240.9271393, 696.1613770, -252.9785004, 729.6998291, -970.6269531, 949.1398315

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818314, upper bound: 743.6819934
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817848, upper bound: 743.6818239
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -112.5535049, 587.8195190, -245.0853882, 1306.8864746, -1419.4399414, 832.9049072
1: -183.7729950, 697.9183960, -402.3925171, 1551.9332275, -1735.7060547, 1100.3109131
2: -129.9520111, 722.9826050, -284.3505859, 1604.2086182, -1734.1606445, 1007.3331909
3: -316.7823486, 611.9615479, -696.2509766, 1362.7196045, -1679.5018311, 1308.2125244
4: -213.9270020, 619.8825684, -469.2285461, 1376.6607666, -1590.5877686, 1089.1110840

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786434, upper bound: 743.6778042
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6614798
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6512203, upper bound: 743.6509690
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -126.3022079, 661.6420898, -245.0362701, 1306.6394043, -1432.9412842, 906.6782837
1: -205.2046204, 785.4786377, -402.3119507, 1551.6401367, -1756.8447266, 1187.7905273
2: -145.9328613, 813.9240112, -284.2939453, 1603.9042969, -1749.8371582, 1098.2180176
3: -355.0084839, 687.8810425, -696.1120605, 1362.4587402, -1717.4671631, 1383.9929199
4: -240.9271393, 696.1613770, -469.1342773, 1376.3970947, -1617.3238525, 1165.2956543

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786434, upper bound: 743.6777250
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6608289
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6512203, upper bound: 743.6509690
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -219.2709503, 1174.1640625, -122.8302841, 645.9752808, -865.2460938, 1296.9940186
1: -361.2863770, 1393.5151367, -199.6369781, 765.9796143, -1127.2658691, 1593.1518555
2: -254.5155334, 1441.5292969, -142.1338348, 795.0182495, -1049.5338135, 1583.6630859
3: -624.3097534, 1221.4396973, -345.8247986, 669.5960693, -1293.9057617, 1567.2641602
4: -419.5296021, 1235.2445068, -234.5676575, 679.1900024, -1098.7194824, 1469.8120117

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6765584, upper bound: 743.6776218
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762959, upper bound: 743.6770307
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -222.4794769, 1191.5399170, -123.4631195, 649.6119385, -872.0913696, 1315.0028076
1: -366.2840271, 1414.3809814, -200.7268982, 770.2991333, -1136.5830078, 1615.1079102
2: -258.2010803, 1462.7208252, -142.8895416, 799.4918213, -1057.6928711, 1605.6102295
3: -633.0128174, 1239.9426270, -347.7039185, 673.3336182, -1306.3464355, 1587.6464844
4: -425.6152649, 1253.3726807, -235.8223267, 682.9373779, -1108.5526123, 1489.1949463

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -218.3511963, 1169.3743896, -126.5166855, 664.8497314, -883.2009277, 1295.8908691
1: -359.7224426, 1387.7816162, -205.7110748, 788.4483643, -1148.1705322, 1593.4926758
2: -253.4371948, 1435.6456299, -146.4772949, 818.3582764, -1071.7954102, 1582.1228027
3: -621.5870361, 1216.3245850, -356.1877441, 689.3673706, -1310.9543457, 1572.5123291
4: -417.7348938, 1230.1245117, -241.7033234, 699.3020020, -1117.0364990, 1471.8278809

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6759333, upper bound: 743.6764066
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761382, upper bound: 743.6765961
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -221.6005402, 1186.9974365, -127.1879044, 668.6716919, -890.2721558, 1314.1849365
1: -364.7952271, 1408.9403076, -206.8559265, 792.9954834, -1157.7905273, 1615.7962646
2: -257.1717834, 1457.1420898, -147.2743683, 823.0498047, -1080.2215576, 1604.4163818
3: -630.4287109, 1235.0737305, -358.1561890, 693.3303833, -1323.7590332, 1593.2298584
4: -423.8996582, 1248.5048828, -243.0264130, 703.2523193, -1127.1519775, 1491.5312500

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6765722, upper bound: 743.6766788
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.23 seconds
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6830842, upper bound: 743.6821182
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6820360, upper bound: 743.6819924
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6830842, upper bound: 743.6821284
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6820360, upper bound: 743.6819924
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6608289
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6537467, upper bound: 743.6526691
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6608289
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6537467, upper bound: 743.6526691
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6818314, upper bound: 743.6819892
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6817848, upper bound: 743.6818247
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6818314, upper bound: 743.6819934
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6817848, upper bound: 743.6818239
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6614798
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6512203, upper bound: 743.6509690
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6604131, upper bound: 743.6608289
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6512203, upper bound: 743.6509690
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6765584, upper bound: 743.6776218
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6762959, upper bound: 743.6770307
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6770791, upper bound: 743.6780407
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6759333, upper bound: 743.6764066
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6761382, upper bound: 743.6765961
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.23
Output dim: 0, lower bound: -743.6765722, upper bound: 743.6766788

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.9838715, 579.2432861, -105.9339447, 552.7522583, -663.7358398, 685.1772461
1: -181.1829071, 687.7188110, -173.2981720, 656.3524170, -837.5352173, 861.0169678
2: -128.1228180, 712.4804077, -122.4249039, 680.0018311, -808.1245728, 834.9053345
3: -312.3287964, 602.9791260, -298.7279358, 575.1688843, -887.4976807, 901.7070312
4: -210.8973389, 610.8708496, -201.5428467, 582.7217407, -793.6190796, 812.4136963

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818616
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830089, upper bound: 743.6818928
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -109.9911194, 574.4713135, -110.8992386, 579.6266479, -689.6177368, 685.3705444
1: -179.5732117, 681.9700317, -181.2997894, 688.1677856, -867.7409058, 863.2698364
2: -126.9732590, 706.6024780, -128.1793365, 713.0130615, -839.9862671, 834.7817993
3: -309.4883423, 597.7264404, -312.7931824, 603.0380249, -912.5263672, 910.5196533
4: -208.9729309, 605.6107788, -211.1096497, 610.7996826, -819.7725220, 816.7203369

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6822792, upper bound: 743.6819573
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6819573, upper bound: 743.6819573
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -124.7146759, 653.2996826, -105.9339447, 552.7522583, -677.4668579, 759.2336426
1: -202.5810089, 775.5338745, -173.2981720, 656.3524170, -858.9334106, 948.8320312
2: -144.0865784, 803.6813965, -122.4249039, 680.0018311, -824.0883789, 926.1063232
3: -350.5056763, 679.0130005, -298.7279358, 575.1688843, -925.6744995, 977.7407837
4: -237.8714752, 687.3131714, -201.5428467, 582.7217407, -820.5932007, 888.8560181

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6731041, upper bound: 743.6710495
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776699, upper bound: 743.6763207
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6524395, upper bound: 743.6534130
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -124.0691986, 649.7680054, -110.8992386, 579.6266479, -703.6958618, 760.6672363
1: -201.5406342, 771.3380737, -181.2997894, 688.1677856, -889.7083740, 952.6378784
2: -143.3399048, 799.3686523, -128.1793365, 713.0130615, -856.3529053, 927.5479736
3: -348.6892700, 675.3395996, -312.7931824, 603.0380249, -951.7272949, 988.1328125
4: -236.6308594, 683.5281372, -211.1096497, 610.7996826, -847.4305420, 894.6377563

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6753760, upper bound: 743.6740579
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6505715, upper bound: 743.6470422
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6373981, upper bound: 743.6365444
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.9838715, 579.2432861, -127.9071960, 668.6610718, -779.6447144, 707.1505127
1: -181.1829071, 687.7188110, -207.9375000, 794.0432129, -975.2259521, 895.6563110
2: -128.1228180, 712.4804077, -147.8359528, 822.7031860, -950.8259277, 860.3162842
3: -312.3287964, 602.9791260, -359.8731689, 695.8218384, -1008.1506348, 962.8522949
4: -210.8973389, 610.8708496, -244.1461182, 704.1927490, -915.0900879, 855.0169678

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6674925, upper bound: 743.6671376
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801696, upper bound: 743.6798444
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -109.9911194, 574.4713135, -133.5231628, 696.6735840, -806.6645508, 707.9945068
1: -179.5732117, 681.9700317, -216.7913666, 827.4710693, -1007.0441895, 898.7614136
2: -126.9732590, 706.6024780, -154.2804565, 857.2047119, -984.1779175, 860.8829346
3: -309.4883423, 597.7264404, -375.3381958, 725.5678101, -1035.0557861, 973.0646362
4: -208.9729309, 605.6107788, -254.9369507, 733.7011719, -942.6740723, 860.5476685

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6669199, upper bound: 743.6671003
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800618, upper bound: 743.6799114
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -124.7146759, 653.2996826, -127.9071960, 668.6610718, -793.3756714, 781.2069092
1: -202.5810089, 775.5338745, -207.9375000, 794.0432129, -996.6242065, 983.4713135
2: -144.0865784, 803.6813965, -147.8359528, 822.7031860, -966.7897339, 951.5172729
3: -350.5056763, 679.0130005, -359.8731689, 695.8218384, -1046.3273926, 1038.8861084
4: -237.8714752, 687.3131714, -244.1461182, 704.1927490, -942.0642090, 931.4592896

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6710910, upper bound: 743.6702653
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801008, upper bound: 743.6799535
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -124.0691986, 649.7680054, -133.5231628, 696.6735840, -820.7426758, 783.2911377
1: -201.5406342, 771.3380737, -216.7913666, 827.4710693, -1029.0117188, 988.1294556
2: -143.3399048, 799.3686523, -154.2804565, 857.2047119, -1000.5445557, 953.6491089
3: -348.6892700, 675.3395996, -375.3381958, 725.5678101, -1074.2565918, 1050.6777344
4: -236.6308594, 683.5281372, -254.9369507, 733.7011719, -970.3320312, 938.4650879

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708230, upper bound: 743.6700693
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798297, upper bound: 743.6798407
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -217.1668243, 1163.3778076, -118.4671173, 624.0725708, -841.2393799, 1281.8448486
1: -357.9055786, 1380.6646729, -192.7066345, 739.8546143, -1097.7602539, 1573.3712158
2: -252.0911407, 1428.2781982, -137.1858368, 768.1466675, -1020.2377930, 1565.4639893
3: -618.4486694, 1210.0700684, -333.8510132, 646.2940063, -1264.7426758, 1543.9211426
4: -415.5505981, 1223.8265381, -226.3930511, 655.9114380, -1071.4620361, 1450.2194824

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732408, upper bound: 743.6736001
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708673, upper bound: 743.6714788
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -217.8907166, 1167.5141602, -131.3227234, 692.5329590, -910.4237061, 1298.8366699
1: -359.1358948, 1385.5222168, -214.0601044, 821.1294556, -1180.2650146, 1599.5821533
2: -252.9543610, 1433.3623047, -152.0366974, 852.3486938, -1105.3031006, 1585.3988037
3: -620.5607300, 1214.2648926, -370.3776855, 717.2387085, -1337.7993164, 1584.6423340
4: -416.9497070, 1228.0989990, -250.6942749, 727.6430054, -1144.5927734, 1478.7932129

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739132, upper bound: 743.6745318
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754525, upper bound: 743.6759131
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -222.4794769, 1191.5399170, -119.6911011, 629.6958008, -852.1751709, 1311.2308350
1: -366.2840271, 1414.3809814, -194.4678040, 746.5281982, -1112.8118896, 1608.8487549
2: -258.2010803, 1462.7208252, -138.4603119, 775.1241455, -1033.3251953, 1601.1811523
3: -633.0128174, 1239.9426270, -336.9168091, 652.2282104, -1285.2409668, 1576.8593750
4: -425.6152649, 1253.3726807, -228.4606628, 661.8424683, -1087.4577637, 1481.8330078

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708992, upper bound: 743.6714760
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -222.4794769, 1191.5399170, -121.6490402, 640.5430908, -863.0224609, 1313.1887207
1: -366.2840271, 1414.3809814, -197.9536591, 759.4238281, -1125.7075195, 1612.3344727
2: -258.2010803, 1462.7208252, -140.8072357, 788.4388428, -1046.6398926, 1603.5280762
3: -633.0128174, 1239.9426270, -342.8246765, 663.4189453, -1296.4317627, 1582.7672119
4: -425.6152649, 1253.3726807, -232.2611084, 673.3063354, -1098.9216309, 1485.6337891

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708992, upper bound: 743.6714760
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -216.2361755, 1158.5341797, -122.6900330, 645.5486450, -861.7847900, 1281.2239990
1: -356.3269043, 1374.8645020, -199.6589661, 765.4428711, -1121.7697754, 1574.5234375
2: -251.0007324, 1422.3310547, -142.1393585, 794.6860962, -1045.6867676, 1564.4703369
3: -615.7015381, 1204.8913574, -345.7126465, 668.8844604, -1284.5859375, 1550.6040039
4: -413.7331543, 1218.6495361, -234.5329437, 678.8283691, -1092.5612793, 1453.1823730

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6762386
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6764066
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -216.9475861, 1162.5975342, -134.9858398, 711.2059937, -928.1535034, 1297.5832520
1: -357.5334473, 1379.6369629, -220.1028900, 843.3987427, -1200.9318848, 1599.7398682
2: -251.8485718, 1427.3226318, -156.3443298, 875.4148560, -1127.2634277, 1583.6669922
3: -617.7713013, 1209.0170898, -380.6785278, 737.0870972, -1354.8582764, 1589.6953125
4: -415.1094360, 1222.8458252, -257.7850342, 747.6534424, -1162.7629395, 1480.6307373

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6730832, upper bound: 743.6734441
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6707200, upper bound: 743.6712915
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -219.4888611, 1176.2104492, -123.3759003, 649.4578857, -868.9467163, 1299.5863037
1: -361.3711243, 1396.0882568, -200.8105927, 770.0906982, -1131.4615479, 1596.8988037
2: -254.7278137, 1443.8673096, -142.9484253, 799.4795532, -1054.2073975, 1586.8156738
3: -624.4890747, 1223.6979980, -347.7085571, 672.9373779, -1297.4262695, 1571.4064941
4: -419.8954468, 1237.0604248, -235.8834381, 682.8620605, -1102.7575684, 1472.9438477

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -220.3145294, 1180.8150635, -136.1291046, 717.7438965, -938.0584106, 1316.9440918
1: -362.7955322, 1401.5093994, -222.0205994, 851.1801147, -1213.9754639, 1623.5299072
2: -255.7187042, 1449.5502930, -157.6913605, 883.3920898, -1139.1107178, 1607.2416992
3: -626.9431763, 1228.4073486, -383.9887085, 743.9482422, -1370.8913574, 1612.3959961
4: -421.4999390, 1241.8664551, -260.0278320, 754.4224854, -1175.9223633, 1501.8940430

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6735091, upper bound: 743.6736278
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6707553, upper bound: 743.6712848
time: 0.55 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.26 seconds
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818616
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6830089, upper bound: 743.6818928
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6822792, upper bound: 743.6819573
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6819573, upper bound: 743.6819573
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6776699, upper bound: 743.6763207
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6524395, upper bound: 743.6534130
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6505715, upper bound: 743.6470422
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6373981, upper bound: 743.6365444
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6674925, upper bound: 743.6671376
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6801696, upper bound: 743.6798444
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6669199, upper bound: 743.6671003
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6800618, upper bound: 743.6799114
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6710910, upper bound: 743.6702653
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6801008, upper bound: 743.6799535
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6708230, upper bound: 743.6700693
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6798297, upper bound: 743.6798407
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6732408, upper bound: 743.6736001
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6708673, upper bound: 743.6714788
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6739132, upper bound: 743.6745318
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6754525, upper bound: 743.6759131
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6708992, upper bound: 743.6714760
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6708992, upper bound: 743.6714760
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6762386
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6764066
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6730832, upper bound: 743.6734441
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6707200, upper bound: 743.6712915
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6735091, upper bound: 743.6736278
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.26
Output dim: 0, lower bound: -743.6707553, upper bound: 743.6712848

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -101.0690155, 526.4794312, -103.7301712, 541.3041992, -642.3731689, 630.2094727
1: -164.8477631, 624.9663086, -169.7471771, 642.6962280, -807.5439453, 794.7134399
2: -116.5611420, 647.6973267, -119.8878021, 665.9727783, -782.5339355, 767.5851440
3: -284.2959290, 547.8026123, -292.5841980, 563.0449219, -847.3408203, 840.3868408
4: -191.8051147, 555.3035889, -197.3354340, 570.5695801, -762.3746338, 752.6390381

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807482, upper bound: 743.6799384
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818613
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -105.8922195, 552.6226196, -103.3648453, 539.3392334, -645.2313843, 655.9874878
1: -172.8340149, 655.9475708, -169.0999603, 640.3439331, -813.1779785, 825.0474854
2: -122.2281647, 679.8639526, -119.4540939, 663.5599365, -785.7880859, 799.3180542
3: -297.9086304, 574.8199463, -291.4603577, 560.9758301, -858.8844604, 866.2802734
4: -201.1273651, 582.6086426, -196.6144562, 568.4857788, -769.6131592, 779.2229614

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6691479, upper bound: 743.6670051
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6671817, upper bound: 743.6653825
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -100.2289200, 522.4281006, -108.7663193, 568.5158081, -668.7446899, 631.1943970
1: -163.4556427, 620.1143799, -177.8553009, 674.9139404, -838.3695679, 797.9696655
2: -115.5788956, 642.6743164, -125.7270508, 699.3925781, -814.9714966, 768.4013672
3: -281.8478699, 543.3995361, -306.8491516, 591.2888184, -873.1366577, 850.2486572
4: -190.1617737, 550.8408203, -207.0492554, 599.0328979, -789.1947021, 757.8900757

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776177, upper bound: 743.6779100
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776935, upper bound: 743.6772641
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -104.8915634, 547.8166504, -108.2731476, 565.9613647, -670.8528442, 656.0897827
1: -171.2113953, 650.1622314, -177.0132904, 671.8518066, -843.0632324, 827.1754150
2: -121.0690231, 673.9449463, -125.1451874, 696.2468872, -817.3159180, 799.0900879
3: -295.0421143, 569.5279541, -305.3728638, 588.5610962, -883.6030884, 874.9008179
4: -199.1848602, 577.3074951, -206.0783081, 596.2843628, -795.4691772, 783.3858032

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775801, upper bound: 743.6784338
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776527, upper bound: 743.6776527
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -123.1612854, 645.3696289, -105.8333206, 552.2373657, -675.3986816, 751.2029419
1: -200.0756683, 766.0589600, -173.1361237, 655.7371826, -855.8128662, 939.1950684
2: -142.3113556, 793.9484863, -122.3097763, 679.3688965, -821.6802368, 916.2582397
3: -346.1917419, 670.5980835, -298.4485474, 574.6245728, -920.8162842, 969.0466309
4: -234.9388428, 678.9008789, -201.3526459, 582.1762695, -817.1151123, 880.2534790

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6524395, upper bound: 743.6534130
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6524395, upper bound: 743.6534130
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -108.1043472, 565.9071655, -125.2255936, 654.1320190, -762.2363281, 691.1326904
1: -176.2869873, 671.7727661, -203.5917664, 776.7348022, -953.0216675, 875.3645020
2: -124.7286987, 695.7778320, -144.7081451, 804.9599609, -929.6886597, 840.4859619
3: -303.7583313, 588.4263916, -352.3194580, 680.4817505, -984.2399902, 940.7458496
4: -205.2391663, 595.8842163, -238.8982086, 688.9557495, -894.1949463, 834.7824097

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6628064, upper bound: 743.6628059
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799225, upper bound: 743.6797955
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798252, upper bound: 743.6795410
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -107.4235153, 562.6148071, -130.8608704, 682.3421021, -789.7655029, 693.4757080
1: -175.1909332, 667.8115234, -212.5088654, 810.4471436, -985.6380615, 880.3203735
2: -123.9438477, 691.7254028, -151.1832428, 839.7044678, -963.6482544, 842.9086304
3: -301.8308411, 584.8240356, -367.8644104, 710.5735474, -1012.4044189, 952.6884766
4: -203.9245453, 592.2894897, -249.7551727, 718.7129517, -922.6374512, 842.0446777

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6727747, upper bound: 743.6740953
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798062, upper bound: 743.6797369
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798060, upper bound: 743.6797406
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -122.2582626, 640.5799561, -125.2255936, 654.1320190, -776.3901978, 765.8054810
1: -198.5961761, 760.3625488, -203.5917664, 776.7348022, -975.3309326, 963.9542847
2: -141.2095184, 788.1031494, -144.7081451, 804.9599609, -946.1694336, 932.8112793
3: -343.4842224, 665.6071777, -352.3194580, 680.4817505, -1023.9658203, 1017.9266357
4: -232.9744415, 673.7650146, -238.8982086, 688.9557495, -921.9301147, 912.6632080

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799480, upper bound: 743.6798741
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797615, upper bound: 743.6795819
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -121.7903671, 638.1036377, -130.8608704, 682.3421021, -804.1323242, 768.9644165
1: -197.8604736, 757.4287720, -212.5088654, 810.4471436, -1008.3076172, 969.9376221
2: -140.6688080, 785.0625610, -151.1832428, 839.7044678, -980.3732910, 936.2457886
3: -342.1788635, 663.0921631, -367.8644104, 710.5735474, -1052.7524414, 1030.9562988
4: -232.0746307, 671.0882568, -249.7551727, 718.7129517, -950.7875977, 920.8433838

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797102, upper bound: 743.6797434
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797102, upper bound: 743.6797323
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -211.3190613, 1132.3103027, -129.2078400, 681.3307495, -892.6497803, 1261.5180664
1: -348.4545593, 1343.4644775, -210.5583344, 807.7537842, -1156.2083740, 1554.0224609
2: -245.3459625, 1390.3310547, -149.5817871, 838.6286011, -1083.9746094, 1539.9128418
3: -602.0980225, 1176.9211426, -364.3678589, 705.3854980, -1307.4835205, 1541.2890625
4: -404.3329468, 1190.9315186, -246.6425476, 715.8007812, -1120.1335449, 1437.5737305

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739132, upper bound: 743.6745318
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739132, upper bound: 743.6745318
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -216.4500427, 1160.1343994, -122.7716904, 648.1203003, -864.5703125, 1282.9061279
1: -356.9416809, 1376.2165527, -200.1265869, 767.8892212, -1124.8308105, 1576.3430176
2: -251.4054871, 1424.5469971, -142.1008911, 797.9393921, -1049.3448486, 1566.6479492
3: -616.9926758, 1204.9918213, -346.1785583, 669.5283203, -1286.5209961, 1551.1702881
4: -414.1963196, 1219.5759277, -234.1398621, 680.0571289, -1094.2534180, 1453.7158203

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754525, upper bound: 743.6759131
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754525, upper bound: 743.6759131
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -223.0012970, 1193.5424805, -118.8030090, 625.0287476, -848.0299072, 1312.3453369
1: -366.8967285, 1416.7996826, -193.0054321, 740.9705200, -1107.8671875, 1609.8050537
2: -258.7090759, 1465.1553955, -137.4347687, 769.3919678, -1028.1009521, 1602.5902100
3: -634.0860596, 1242.1962891, -334.4058533, 647.3474731, -1281.4329834, 1576.6020508
4: -426.4082947, 1255.6754150, -226.7731934, 656.9318237, -1083.3398438, 1482.4486084

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749040, upper bound: 743.6757584
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -223.0012970, 1193.5424805, -120.5703812, 634.8587036, -857.8598633, 1314.1129150
1: -366.8967285, 1416.7996826, -196.1848755, 752.6387939, -1119.5355225, 1612.9846191
2: -258.7090759, 1465.1553955, -139.5618439, 781.4649048, -1040.1739502, 1604.7170410
3: -634.0860596, 1242.1962891, -339.7887878, 657.4399414, -1291.5258789, 1581.9851074
4: -426.4082947, 1255.6754150, -230.2015839, 667.3248901, -1093.7330322, 1485.8769531

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -210.0517578, 1124.9890137, -122.6900330, 645.5486450, -855.6002808, 1247.6788330
1: -346.2947388, 1335.0095215, -199.6589661, 765.4428711, -1111.7375488, 1534.6684570
2: -243.7780151, 1381.2087402, -142.1393585, 794.6860962, -1038.4641113, 1523.3480225
3: -598.4502563, 1169.9744873, -345.7126465, 668.8844604, -1267.3347168, 1515.6871338
4: -401.7911987, 1183.5435791, -234.5329437, 678.8283691, -1080.6195068, 1418.0765381

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750196, upper bound: 743.6762386
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6762386
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -214.2097473, 1147.7625732, -122.6900330, 645.5486450, -859.7583008, 1270.4526367
1: -353.0095825, 1362.0548096, -199.6589661, 765.4428711, -1118.4523926, 1561.7137451
2: -248.6430969, 1409.1303711, -142.1393585, 794.6860962, -1043.3292236, 1551.2696533
3: -609.9364624, 1193.5659180, -345.7126465, 668.8844604, -1278.8209229, 1539.2784424
4: -409.8286743, 1207.2530518, -234.5329437, 678.8283691, -1088.6569824, 1441.7860107

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6764066
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6764066
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -213.3117828, 1143.1403809, -123.3759003, 649.4578857, -862.7696533, 1266.5162354
1: -351.4211426, 1356.6577148, -200.8105927, 770.0906982, -1121.5115967, 1557.4682617
2: -247.5558624, 1403.3599854, -142.9484253, 799.4795532, -1047.0354004, 1546.3083496
3: -607.3508301, 1188.9392090, -347.7085571, 672.9373779, -1280.2882080, 1536.6477051
4: -407.9915466, 1202.2954102, -235.8834381, 682.8620605, -1090.8536377, 1438.1788330

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -217.4695587, 1165.4942627, -123.3759003, 649.4578857, -866.9274292, 1288.8701172
1: -358.0655212, 1383.3413086, -200.8105927, 770.0906982, -1128.1561279, 1584.1518555
2: -252.3796692, 1430.7321777, -142.9484253, 799.4795532, -1051.8592529, 1573.6806641
3: -618.7449951, 1212.4259033, -347.7085571, 672.9373779, -1291.6821289, 1560.1342773
4: -416.0061340, 1225.7145996, -235.8834381, 682.8620605, -1098.8680420, 1461.5980225

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
time: 0.55 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.29 seconds
NS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6807482, upper bound: 743.6799384
NS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818613
NS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6691479, upper bound: 743.6670051
NS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6671817, upper bound: 743.6653825
NS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6776177, upper bound: 743.6779100
NS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6776935, upper bound: 743.6772641
NS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6775801, upper bound: 743.6784338
NS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6776527, upper bound: 743.6776527
NS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6524395, upper bound: 743.6534130
NS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6524395, upper bound: 743.6534130
NS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6799225, upper bound: 743.6797955
NS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6798252, upper bound: 743.6795410
NS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6798062, upper bound: 743.6797369
NS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6798060, upper bound: 743.6797406
NS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6799480, upper bound: 743.6798741
NS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6797615, upper bound: 743.6795819
NS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6797102, upper bound: 743.6797434
NS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6797102, upper bound: 743.6797323
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6739132, upper bound: 743.6745318
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6739132, upper bound: 743.6745318
NS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6754525, upper bound: 743.6759131
NS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6754525, upper bound: 743.6759131
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6749040, upper bound: 743.6757584
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6750196, upper bound: 743.6762386
NS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6762386
NS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6764066
NS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6750590, upper bound: 743.6764066
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
NS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655
NS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.29
Output dim: 0, lower bound: -743.6763700, upper bound: 743.6765655

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -98.4522171, 512.8218994, -98.4417725, 513.5307617, -611.9829712, 611.2636719
1: -160.5675201, 608.6746826, -161.1413116, 609.5267944, -770.0942383, 769.8159790
2: -113.5348892, 630.9464722, -113.7735596, 631.9451294, -745.4800415, 744.7200317
3: -276.9085388, 533.3290405, -277.7285461, 533.5625000, -810.4710693, 811.0576172
4: -186.7921448, 540.8135986, -187.1896667, 541.1702881, -727.9622803, 728.0032349

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695808, upper bound: 743.6692888
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807482, upper bound: 743.6799384
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807482, upper bound: 743.6799384
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -99.4164200, 518.0441895, -100.9430389, 527.7259521, -627.1421509, 618.9872437
1: -162.2023773, 614.9038696, -165.1048584, 626.0836182, -788.2859497, 780.0087280
2: -114.6692657, 637.3526611, -116.6850052, 649.6621704, -764.3314209, 754.0375366
3: -279.7018738, 538.8558960, -284.7823181, 548.0885620, -827.7904053, 823.6381836
4: -188.6690674, 546.3233032, -192.0881348, 555.7432251, -744.4122925, 738.4114380

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818613
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818613
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -100.2289200, 522.4281006, -103.9064178, 542.6088257, -642.8377075, 626.3345337
1: -163.4556427, 620.1143799, -169.6931152, 644.1962891, -807.6519165, 789.8074951
2: -115.5788956, 642.6743164, -119.9905090, 667.4776611, -783.0565796, 762.6647949
3: -281.8478699, 543.3995361, -292.6869202, 564.4006348, -846.2485352, 836.0864258
4: -190.1617737, 550.8408203, -197.5927582, 571.8790894, -762.0408325, 748.4335938

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775950, upper bound: 743.6772422
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775950, upper bound: 743.6772641
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -97.2843399, 506.9355164, -107.3521576, 561.1909790, -658.4753418, 614.2876587
1: -158.6568146, 601.6524048, -175.4548340, 666.0117798, -824.6685181, 777.1072388
2: -112.1638718, 623.6851196, -124.0597000, 690.5047607, -802.6686401, 747.7448120
3: -273.5513916, 527.0639038, -302.6736755, 583.1193848, -856.6707153, 829.7375488
4: -184.5118103, 534.4744263, -204.2476501, 591.0522461, -775.5640869, 738.7218628

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776688, upper bound: 743.6764351
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776935, upper bound: 743.6772641
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -104.8915634, 547.8166504, -103.2229309, 539.0735474, -643.9650879, 651.0395508
1: -171.2113953, 650.1622314, -168.5424194, 639.9636841, -811.1750488, 818.7045288
2: -121.0690231, 673.9449463, -119.1834106, 663.1408691, -784.2098999, 793.1282959
3: -295.0421143, 569.5279541, -290.6762390, 560.6245728, -855.6665649, 860.2042236
4: -199.1848602, 577.3074951, -196.2503967, 568.0711670, -767.2559204, 773.5578613

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6625620, upper bound: 743.6633619
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6591289, upper bound: 743.6593022
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -102.2290192, 533.7502441, -107.0489426, 559.6320190, -661.8609009, 640.7990723
1: -166.8485718, 633.4045410, -174.9123230, 664.1235962, -830.9720459, 808.3168335
2: -117.9771957, 656.6885376, -123.7044678, 688.5852661, -806.5624390, 780.3930054
3: -287.5153503, 554.7431641, -301.7388306, 581.4338379, -868.9491577, 856.4819946
4: -194.0705261, 562.4859009, -203.6540527, 589.3750000, -783.4455566, 766.1399536

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6622855, upper bound: 743.6627982
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6579017, upper bound: 743.6579017
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -105.9468002, 554.5289307, -115.7844238, 604.9573975, -710.9041748, 670.3133545
1: -172.7748566, 658.2072144, -188.0888214, 718.2243042, -890.9990845, 846.2960205
2: -122.2322083, 681.8568115, -133.7082825, 744.5399170, -866.7720947, 815.5650635
3: -297.6916809, 576.4096680, -325.6826172, 628.7434082, -926.4349976, 902.0922852
4: -201.1029358, 583.8429565, -220.7799072, 636.7697754, -837.8726807, 804.6228638

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6686914, upper bound: 743.6684546
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799225, upper bound: 743.6797955
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.5176773, 552.3764038, -121.3843231, 633.8896484, -739.4072876, 673.7607422
1: -172.0396729, 655.6303711, -197.2928009, 752.6469727, -924.6865845, 852.9231567
2: -121.7332077, 679.2062988, -140.2562103, 780.1546021, -901.8878174, 819.4625244
3: -296.4242249, 574.1198730, -341.4319763, 659.1729126, -955.5971680, 915.5518188
4: -200.2779999, 581.5245972, -231.5281372, 667.5231934, -867.8011475, 813.0527344

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6713030, upper bound: 743.6720681
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6643907, upper bound: 743.6637235
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -105.3204193, 551.5218506, -121.6504211, 634.0029297, -739.3233032, 673.1722412
1: -171.7693481, 654.5877075, -197.3495178, 752.8806763, -924.6499634, 851.9371338
2: -121.5125580, 678.1530762, -140.4354858, 780.3371582, -901.8497314, 818.5885620
3: -295.9212036, 573.1154785, -341.8585205, 659.6485596, -955.5697632, 914.9739990
4: -199.8964844, 580.5571899, -232.0285339, 667.4777222, -867.3742065, 812.5856323

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6789944, upper bound: 743.6791043
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6789165, upper bound: 743.6790851
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -104.7507935, 548.6486816, -127.1046982, 662.3773804, -767.1281128, 675.7533569
1: -170.8016357, 651.1483765, -206.3558960, 786.6911621, -957.4927979, 857.5042114
2: -120.8469849, 674.6199951, -146.8322754, 815.2750854, -936.1220093, 821.4522095
3: -294.2497253, 570.0490723, -357.2191467, 689.5650635, -983.8148193, 927.2680664
4: -198.7944183, 577.4577637, -242.5505524, 697.6455078, -896.4397583, 820.0083008

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777250, upper bound: 743.6781987
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798060, upper bound: 743.6797406
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -120.2517471, 629.9382324, -115.7844238, 604.9573975, -725.2091064, 745.7225342
1: -195.3569641, 747.7091675, -188.0888214, 718.2243042, -913.5811768, 935.7979126
2: -138.8953400, 775.0850830, -133.7082825, 744.5399170, -883.4352417, 908.7933350
3: -337.8917542, 654.3988647, -325.6826172, 628.7434082, -966.6351318, 980.0814819
4: -229.1430817, 662.5228271, -220.7799072, 636.7697754, -865.9128418, 883.3027344

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6720584, upper bound: 743.6713518
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6640244, upper bound: 743.6623264
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -119.9591904, 628.5148315, -121.3843231, 633.8896484, -753.8487549, 749.8991699
1: -194.8238831, 745.9780884, -197.2928009, 752.6469727, -947.4708252, 943.2708740
2: -138.5514221, 773.3291016, -140.2562103, 780.1546021, -918.7060547, 913.5853271
3: -336.9652710, 652.8312378, -341.4319763, 659.1729126, -996.1380615, 994.2631836
4: -228.5690308, 660.9846191, -231.5281372, 667.5231934, -896.0921631, 892.5127563

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745499, upper bound: 743.6742331
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6692760, upper bound: 743.6677868
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -119.7447128, 627.2639771, -121.6504211, 634.0029297, -753.7476196, 748.9144287
1: -194.5491791, 744.5421143, -197.3495178, 752.8806763, -947.4296875, 941.8915405
2: -138.3079834, 771.7946777, -140.4354858, 780.3371582, -918.6451416, 912.2301636
3: -336.4696045, 651.6750488, -341.8585205, 659.6485596, -996.1181641, 993.5335083
4: -228.1666107, 659.6263428, -232.0285339, 667.4777222, -895.6443481, 891.6547852

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746849, upper bound: 743.6752246
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797102, upper bound: 743.6797434
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -119.5027618, 626.0648193, -127.1046982, 662.3773804, -781.8801270, 753.1693726
1: -194.1037445, 743.0811768, -206.3558960, 786.6911621, -980.7949219, 949.4370728
2: -138.0220947, 770.3205566, -146.8322754, 815.2750854, -953.2971802, 917.1527710
3: -335.6885986, 650.3587036, -357.2191467, 689.5650635, -1025.2535400, 1007.5777588
4: -227.6882782, 658.3469849, -242.5505524, 697.6455078, -925.3335571, 900.8975220

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6697501, upper bound: 743.6709335
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6652939, upper bound: 743.6653703
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -211.3190613, 1132.3103027, -125.1875458, 660.1900635, -871.5091553, 1257.4978027
1: -348.4545593, 1343.4644775, -203.8302612, 782.5266113, -1130.9810791, 1547.2946777
2: -245.3459625, 1390.3310547, -144.8630829, 812.7437134, -1058.0894775, 1535.1940918
3: -602.0980225, 1176.9211426, -352.8000183, 682.9526367, -1285.0505371, 1529.7210693
4: -404.3329468, 1190.9315186, -238.8177795, 693.3311768, -1097.6640625, 1429.7492676

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -211.3190613, 1132.3103027, -128.0259247, 675.7882690, -887.1072998, 1260.3361816
1: -348.4545593, 1343.4644775, -208.8630219, 801.1374512, -1149.5920410, 1552.3273926
2: -245.3459625, 1390.3310547, -148.2407379, 831.8578491, -1077.2037354, 1538.5717773
3: -602.0980225, 1176.9211426, -361.3266907, 699.4045410, -1301.5025635, 1538.2476807
4: -404.3329468, 1190.9315186, -244.3260498, 709.8669434, -1114.1998291, 1435.2572021

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -216.4500427, 1160.1343994, -118.7505341, 626.8072510, -843.2572632, 1278.8848877
1: -356.9416809, 1376.2165527, -193.4060364, 742.4365234, -1099.3780518, 1569.6225586
2: -251.4054871, 1424.5469971, -137.3754883, 771.8521118, -1023.2575684, 1561.9223633
3: -616.9926758, 1204.9918213, -334.6438904, 646.7448730, -1263.7371826, 1539.6357422
4: -414.1963196, 1219.5759277, -226.3007965, 657.4593506, -1071.6556396, 1445.8765869

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748624, upper bound: 743.6759051
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748624, upper bound: 743.6759131
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -216.4500427, 1160.1343994, -121.1776962, 640.7788086, -857.2288208, 1281.3121338
1: -356.9416809, 1376.2165527, -197.7523193, 759.0570679, -1115.9986572, 1573.9688721
2: -251.4054871, 1424.5469971, -140.2706451, 788.9407959, -1040.3463135, 1564.8176270
3: -616.9926758, 1204.9918213, -341.9366760, 661.5886230, -1278.5812988, 1546.9282227
4: -414.1963196, 1219.5759277, -230.9850311, 672.0782471, -1086.2744141, 1450.5607910

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6671377, upper bound: 743.6677136
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748768, upper bound: 743.6755767
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -222.7323914, 1192.2094727, -117.1353607, 613.9246826, -836.6569824, 1309.3446045
1: -366.4578857, 1415.2176514, -190.2110901, 728.4483032, -1094.9060059, 1605.4287109
2: -258.3996277, 1463.5096436, -135.3050385, 755.6836548, -1014.0831909, 1598.8146973
3: -633.3294067, 1240.7879639, -329.4906006, 636.8835449, -1270.2128906, 1570.2785645
4: -425.8945923, 1254.2432861, -223.4666290, 645.3305054, -1071.2250977, 1477.7097168

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -223.0012970, 1193.5424805, -214.9396057, 1145.2023926, -1368.2037354, 1408.4820557
1: -366.8967285, 1416.7996826, -352.7152405, 1358.8963623, -1725.7930908, 1769.5147705
2: -258.7090759, 1465.1553955, -249.2037811, 1406.5161133, -1665.2252197, 1714.3591309
3: -634.0860596, 1242.1962891, -610.8955078, 1191.6590576, -1825.7447510, 1853.0917969
4: -426.4082947, 1255.6754150, -411.1761475, 1206.0214844, -1632.4296875, 1666.8515625

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749040, upper bound: 743.6757584
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -222.7373657, 1192.2343750, -118.8719330, 622.3225098, -845.0598145, 1311.1062012
1: -366.4659729, 1415.2470703, -193.3012085, 738.4765015, -1104.9425049, 1608.5482178
2: -258.4053040, 1463.5400391, -137.3626099, 766.0497437, -1024.4550781, 1600.9025879
3: -633.3433838, 1240.8139648, -334.7113647, 645.7933350, -1279.1367188, 1575.5252686
4: -425.9040833, 1254.2695312, -226.7326508, 654.5278931, -1080.4320068, 1481.0019531

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -223.0012970, 1193.5424805, -220.8299866, 1177.4466553, -1400.4477539, 1414.3724365
1: -366.8967285, 1416.7996826, -362.8426208, 1397.4820557, -1764.3787842, 1779.6423340
2: -258.7090759, 1465.1553955, -256.1611938, 1445.9011230, -1704.6102295, 1721.3166504
3: -634.0860596, 1242.1962891, -628.0755615, 1226.0454102, -1860.1312256, 1870.2717285
4: -426.4082947, 1255.6754150, -422.6713867, 1240.3104248, -1666.7186279, 1678.3468018

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -209.6515656, 1123.0015869, -123.0622025, 644.7520142, -854.4033203, 1246.0635986
1: -345.6375427, 1332.6560059, -200.1659851, 765.1921997, -1110.8294678, 1532.8218994
2: -243.3154297, 1378.7452393, -142.3115997, 793.5733032, -1036.8885498, 1521.0568848
3: -597.3157959, 1167.8820801, -346.4923706, 669.4581909, -1266.7738037, 1514.3743896
4: -401.0236511, 1181.4075928, -234.9912262, 678.2039795, -1079.2275391, 1416.3055420

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6684302, upper bound: 743.6699137
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741761, upper bound: 743.6759539
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -210.0517578, 1124.9890137, -222.7397614, 1187.3519287, -1397.4031982, 1347.7286377
1: -346.2947388, 1335.0095215, -365.8176575, 1409.3018799, -1755.5965576, 1700.8271484
2: -243.7780151, 1381.2087402, -258.4422913, 1458.1481934, -1701.9262695, 1639.6510010
3: -598.4502563, 1169.9744873, -633.2775269, 1236.6293945, -1835.0795898, 1803.2519531
4: -401.7911987, 1183.5435791, -426.5733643, 1251.0395508, -1652.8306885, 1610.1169434

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6684302, upper bound: 743.6699137
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741761, upper bound: 743.6759539
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -213.8079681, 1145.7657471, -123.0622025, 644.7520142, -858.5599365, 1268.8277588
1: -352.3518372, 1359.6884766, -200.1659851, 765.1921997, -1117.5437012, 1559.8543701
2: -248.1792908, 1406.6563721, -142.3115997, 793.5733032, -1041.7524414, 1548.9680176
3: -608.8000488, 1191.4641113, -346.4923706, 669.4581909, -1278.2580566, 1537.9562988
4: -409.0597839, 1205.1063232, -234.9912262, 678.2039795, -1087.2636719, 1439.8347168

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6680217, upper bound: 743.6692390
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6753962, upper bound: 743.6761161
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -214.2097473, 1147.7625732, -222.7397614, 1187.3519287, -1401.5616455, 1370.5023193
1: -353.0095825, 1362.0548096, -365.8176575, 1409.3018799, -1762.3115234, 1727.8724365
2: -248.6430969, 1409.1303711, -258.4422913, 1458.1481934, -1706.7912598, 1667.5726318
3: -609.9364624, 1193.5659180, -633.2775269, 1236.6293945, -1846.5657959, 1826.8431396
4: -409.8286743, 1207.2530518, -426.5733643, 1251.0395508, -1660.8681641, 1633.7696533

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6680217, upper bound: 743.6692390
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6753962, upper bound: 743.6761161
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -213.0381927, 1141.7863770, -123.9678574, 649.6676025, -862.7058105, 1265.7542725
1: -350.9739380, 1355.0524902, -201.6643829, 771.0554810, -1122.0294189, 1556.7166748
2: -247.2403259, 1401.6846924, -143.3663483, 799.5675049, -1046.8078613, 1545.0510254
3: -606.5795288, 1187.5091553, -349.0707703, 674.6644897, -1281.2437744, 1536.5799561
4: -407.4682617, 1200.8381348, -236.7462769, 683.3562622, -1090.8244629, 1437.5842285

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6694391, upper bound: 743.6707601
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6764721, upper bound: 743.6766112
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -213.3117828, 1143.1403809, -223.8315125, 1193.7471924, -1407.0588379, 1366.9718018
1: -351.4211426, 1356.6577148, -367.6460876, 1416.9260254, -1768.3471680, 1724.3038330
2: -247.5558624, 1403.3599854, -259.7287292, 1465.9442139, -1713.5000000, 1663.0887451
3: -607.3508301, 1188.9392090, -636.4033813, 1243.3078613, -1850.6586914, 1825.3425293
4: -407.9915466, 1202.2954102, -428.7227173, 1257.6263428, -1665.6177979, 1631.0180664

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6694391, upper bound: 743.6707601
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6764721, upper bound: 743.6766112
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -217.1943817, 1164.1334229, -123.9678574, 649.6676025, -866.8619995, 1288.1010742
1: -357.6168518, 1381.7265625, -201.6643829, 771.0554810, -1128.6723633, 1583.3907471
2: -252.0628357, 1429.0495605, -143.3663483, 799.5675049, -1051.6302490, 1572.4158936
3: -617.9707642, 1210.9884033, -349.0707703, 674.6644897, -1292.6351318, 1560.0592041
4: -415.4805298, 1224.2508545, -236.7462769, 683.3562622, -1098.8367920, 1460.9969482

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6671995, upper bound: 743.6673131
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6759297, upper bound: 743.6762674
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -217.4695587, 1165.4942627, -223.8315125, 1193.7471924, -1411.2164307, 1389.3256836
1: -358.0655212, 1383.3413086, -367.6460876, 1416.9260254, -1774.9915771, 1750.9874268
2: -252.3796692, 1430.7321777, -259.7287292, 1465.9442139, -1718.3237305, 1690.4609375
3: -618.7449951, 1212.4259033, -636.4033813, 1243.3078613, -1862.0528564, 1848.8291016
4: -416.0061340, 1225.7145996, -428.7227173, 1257.6263428, -1673.6322021, 1654.4372559

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6671995, upper bound: 743.6673131
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6759297, upper bound: 743.6762674
time: 0.63 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.53 seconds
NS_A1_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6807482, upper bound: 743.6799384
NS_A1_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6807482, upper bound: 743.6799384
NS_A1_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818613
NS_A1_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6830170, upper bound: 743.6818613
NS_A1_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6775950, upper bound: 743.6772422
NS_A1_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6775950, upper bound: 743.6772641
NS_A1_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6776688, upper bound: 743.6764351
NS_A1_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6776935, upper bound: 743.6772641
NS_A1_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6625620, upper bound: 743.6633619
NS_A1_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6591289, upper bound: 743.6593022
NS_A1_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6622855, upper bound: 743.6627982
NS_A1_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6579017, upper bound: 743.6579017
NS_A1_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6686914, upper bound: 743.6684546
NS_A1_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6799225, upper bound: 743.6797955
NS_A1_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6713030, upper bound: 743.6720681
NS_A1_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6643907, upper bound: 743.6637235
NS_A1_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6789944, upper bound: 743.6791043
NS_A1_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6789165, upper bound: 743.6790851
NS_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6777250, upper bound: 743.6781987
NS_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6798060, upper bound: 743.6797406
NS_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6720584, upper bound: 743.6713518
NS_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6640244, upper bound: 743.6623264
NS_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6745499, upper bound: 743.6742331
NS_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6692760, upper bound: 743.6677868
NS_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6746849, upper bound: 743.6752246
NS_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6797102, upper bound: 743.6797434
NS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6697501, upper bound: 743.6709335
NS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6652939, upper bound: 743.6653703
NS_A2_B2_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6748624, upper bound: 743.6759051
NS_A2_B2_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6748624, upper bound: 743.6759131
NS_A2_B2_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6671377, upper bound: 743.6677136
NS_A2_B2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6748768, upper bound: 743.6755767
NS_A2_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
NS_A2_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
NS_A2_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
NS_A2_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6749040, upper bound: 743.6757584
NS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
NS_A2_B2_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6684302, upper bound: 743.6699137
NS_A2_B2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6741761, upper bound: 743.6759539
NS_A2_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6684302, upper bound: 743.6699137
NS_A2_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6741761, upper bound: 743.6759539
NS_A2_B2_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6680217, upper bound: 743.6692390
NS_A2_B2_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6753962, upper bound: 743.6761161
NS_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6680217, upper bound: 743.6692390
NS_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6753962, upper bound: 743.6761161
NS_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6694391, upper bound: 743.6707601
NS_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6764721, upper bound: 743.6766112
NS_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6694391, upper bound: 743.6707601
NS_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6764721, upper bound: 743.6766112
NS_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6671995, upper bound: 743.6673131
NS_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6759297, upper bound: 743.6762674
NS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6671995, upper bound: 743.6673131
NS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.53
Output dim: 0, lower bound: -743.6759297, upper bound: 743.6762674

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -95.4706879, 496.6795959, -98.4417725, 513.5307617, -609.0014648, 595.1213379
1: -155.6643677, 589.4772339, -161.1413116, 609.5267944, -765.1910400, 750.6185303
2: -110.0689087, 611.3261719, -113.7735596, 631.9451294, -742.0140381, 725.0997314
3: -268.4704590, 516.3825073, -277.7285461, 533.5625000, -802.0329590, 794.1109619
4: -181.0536957, 523.8262939, -187.1896667, 541.1702881, -722.2239990, 711.0159912

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5999815, upper bound: 743.6034948
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5979911, upper bound: 743.6008566
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.5979911, upper bound: 743.6799384
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -99.9198456, 520.6984253, -98.4417725, 513.5307617, -613.4506226, 619.1401978
1: -162.7367249, 617.8962402, -161.1413116, 609.5267944, -772.2634888, 779.0375366
2: -115.1990280, 640.5969238, -113.7735596, 631.9451294, -747.1441650, 754.3704834
3: -280.8509827, 541.1347656, -277.7285461, 533.5625000, -814.4134521, 818.8632812
4: -189.5567780, 548.5740356, -187.1896667, 541.1702881, -730.7268677, 735.7636719

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5999815, upper bound: 743.6034948
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5979911, upper bound: 743.6008566
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.5979911, upper bound: 743.6799384
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -96.4170761, 501.8118591, -100.9430389, 527.7259521, -624.1430054, 602.7548828
1: -157.2698669, 595.5908203, -165.1048584, 626.0836182, -783.3534546, 760.6956787
2: -111.1826096, 617.5747070, -116.6850052, 649.6621704, -760.8447876, 734.2596436
3: -271.2120667, 521.7926636, -284.7823181, 548.0885620, -819.3006592, 806.5749512
4: -182.8952637, 529.2314453, -192.0881348, 555.7432251, -738.6384888, 721.3195190

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788894, upper bound: 743.6775343
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791889, upper bound: 743.6774775
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -100.8162537, 525.5823975, -100.9430389, 527.7259521, -628.5421143, 626.5254517
1: -164.2793884, 623.7065430, -165.1048584, 626.0836182, -790.3629761, 788.8114014
2: -116.2585754, 646.6009521, -116.6850052, 649.6621704, -765.9206543, 763.2858276
3: -283.4874573, 546.2692871, -284.7823181, 548.0885620, -831.5760498, 831.0516357
4: -191.3052368, 553.7404785, -192.0881348, 555.7432251, -747.0483398, 745.8286133

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788894, upper bound: 743.6775343
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791889, upper bound: 743.6774775
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -95.3979492, 496.6970215, -103.9064178, 542.6088257, -638.0067139, 600.6033325
1: -155.3275909, 589.6099243, -169.6931152, 644.1962891, -799.5237427, 759.3030396
2: -109.8630295, 610.9957886, -119.9905090, 667.4776611, -777.3406982, 730.9862671
3: -267.7618103, 516.7070312, -292.6869202, 564.4006348, -832.1624146, 809.3939209
4: -180.7339325, 523.8508911, -197.5927582, 571.8790894, -752.6129761, 721.4435425

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774488, upper bound: 743.6779100
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -98.2122955, 511.6151428, -103.9064178, 542.6088257, -640.8211060, 615.5215454
1: -160.0363922, 607.0294800, -169.6931152, 644.1962891, -804.2326660, 776.7225952
2: -113.1784210, 629.6752930, -119.9905090, 667.4776611, -780.6560669, 749.6657715
3: -275.9072876, 531.6265869, -292.6869202, 564.4006348, -840.3079224, 824.3134766
4: -186.1045227, 539.2647095, -197.5927582, 571.8790894, -757.9835815, 736.8574219

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -97.0875015, 504.6727295, -106.1155396, 554.7661133, -651.8536377, 610.7882690
1: -158.4272308, 598.8733521, -173.4600067, 658.3429565, -816.7702026, 772.3332520
2: -112.0002899, 621.3995972, -122.6413422, 682.6586914, -794.6589966, 744.0408325
3: -273.0935669, 524.9160767, -299.2316589, 576.3430786, -849.4365234, 824.1477051
4: -184.1421509, 532.4644775, -201.8951721, 584.2666626, -768.4088135, 734.3596191

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774093, upper bound: 743.6764351
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774093, upper bound: 743.6764351
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -96.4520950, 502.6752625, -107.2111969, 560.4737549, -656.9257202, 609.8862915
1: -157.2864990, 596.5634155, -175.2246399, 665.1540527, -822.4405518, 771.7880249
2: -111.2052231, 618.4495239, -123.8978729, 689.6257324, -800.8309326, 742.3474121
3: -271.2001648, 522.5393066, -302.2784729, 582.3577881, -853.5579834, 824.8177490
4: -182.9241180, 529.9328003, -203.9800110, 590.2874146, -773.2114868, 733.9128418

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773991, upper bound: 743.6772636
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773991, upper bound: 743.6772641
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -102.1429138, 534.6565552, -114.0657806, 596.1345825, -698.2774658, 648.7223511
1: -166.5756073, 634.3597412, -185.3272858, 707.6845703, -874.2601929, 819.6869507
2: -117.8728790, 657.7871094, -131.7388458, 733.7158813, -851.5887451, 789.5259399
3: -287.0367432, 555.2524414, -320.9104309, 619.3947754, -906.4315186, 876.1628418
4: -193.8706055, 562.7963257, -217.5302124, 627.3856812, -821.2562866, 780.3264160

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6711248, upper bound: 743.6712606
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6711248, upper bound: 743.6797955
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -86.7550354, 456.9479065, -118.7377548, 618.9193726, -705.6743774, 575.6855469
1: -141.2397003, 541.4791260, -192.5843201, 734.9389648, -876.1786499, 734.0634766
2: -100.0903168, 561.9898071, -137.0914459, 761.7661133, -861.8564453, 699.0812378
3: -243.4175262, 472.5290527, -333.7126160, 643.8546143, -887.2720947, 806.2416382
4: -164.5101929, 479.6804504, -226.5538330, 651.5186157, -816.0288086, 706.2342529

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6654059, upper bound: 743.6647304
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786861, upper bound: 743.6787455
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -103.3944168, 541.4584351, -120.4315186, 627.6745605, -731.0689697, 661.8898315
1: -168.6290588, 642.6998291, -195.3583069, 745.3950806, -914.0241699, 838.0581055
2: -119.3282318, 665.7474976, -139.0473938, 772.5374146, -891.8656616, 804.7947998
3: -290.5924683, 562.7208252, -338.4635010, 653.0977783, -943.6901855, 901.1843262
4: -196.3461761, 569.8480835, -229.7643433, 660.7633057, -857.1093750, 799.6124268

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6664451, upper bound: 743.6666166
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785759, upper bound: 743.6787071
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -99.4692841, 520.9889526, -124.3945694, 648.0044556, -747.4736938, 645.3834839
1: -162.1717529, 618.0921021, -201.9378357, 769.5258179, -931.6974487, 820.0299072
2: -114.7334442, 640.7072754, -143.6992493, 797.6870728, -912.4205322, 784.4064331
3: -279.3399658, 540.6693726, -349.6039429, 674.3285522, -953.6684570, 890.2732544
4: -188.6519623, 548.0852051, -237.3521271, 682.4841309, -871.1361084, 785.4373169

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6646654, upper bound: 743.6653284
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6646654, upper bound: 743.6653284
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -101.0829697, 529.2955933, -125.2825012, 652.9207153, -754.0036011, 654.5780029
1: -164.8126221, 627.9558105, -203.4135284, 775.3942261, -940.2068481, 831.3693237
2: -116.6369171, 651.0010986, -144.7371368, 803.6816406, -920.3185425, 795.7382202
3: -283.9277039, 549.5392456, -352.1448364, 679.5307617, -963.4584961, 901.6840210
4: -191.7958069, 557.0149536, -239.0846100, 687.6041870, -879.4000244, 796.0995483

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6657141, upper bound: 743.6664235
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6657141, upper bound: 743.6797405
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -116.9710159, 613.3043823, -119.5960693, 624.7952881, -741.7661743, 732.9003906
1: -190.0618134, 727.8952026, -194.4413605, 741.8236084, -931.8853760, 922.3365479
2: -135.1426086, 754.6300659, -138.2147980, 768.9725342, -904.1151123, 892.8448486
3: -328.7557373, 636.8291626, -336.5093994, 649.6057129, -978.3613892, 973.3385620
4: -222.9758759, 644.8869629, -228.1727142, 657.8876343, -880.8635254, 873.0596924

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745499, upper bound: 743.6742331
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745499, upper bound: 743.6738001
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -114.8521042, 601.3521729, -119.1648102, 621.0130615, -735.8651733, 720.5169678
1: -186.5377808, 713.6343384, -193.2882385, 737.3674927, -923.9052124, 906.9226074
2: -132.6460724, 740.0847168, -137.5650024, 764.4198608, -897.0659180, 877.6497192
3: -322.6828918, 624.1394653, -334.8556824, 645.8698120, -968.5526733, 958.9951172
4: -218.7908325, 632.2659302, -227.2790833, 653.7066040, -872.4973145, 859.5450439

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6663759, upper bound: 743.6669548
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6663759, upper bound: 743.6749102
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -117.3192902, 615.4348755, -119.9464951, 625.2813721, -742.6006470, 735.3813477
1: -190.3024445, 730.3212891, -194.5947571, 742.4579468, -932.7603760, 924.9160156
2: -135.4558868, 757.1444092, -138.4805908, 769.6211548, -905.0770264, 895.6250000
3: -329.3207397, 638.5756836, -337.1064758, 650.3783569, -979.6989136, 975.6821289
4: -223.5905914, 646.6703491, -228.8014374, 658.1715088, -881.7620850, 875.4717407

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6673739, upper bound: 743.6671685
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793108, upper bound: 743.6793636
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -209.3231812, 1121.8560791, -118.7505341, 626.8072510, -836.1303101, 1240.6064453
1: -345.3283386, 1330.6168213, -193.4060364, 742.4365234, -1087.7648926, 1524.0228271
2: -243.0982056, 1377.6425781, -137.3754883, 771.8521118, -1014.9503174, 1515.0177002
3: -597.0029907, 1164.8403320, -334.6438904, 646.7448730, -1243.7474365, 1499.4842529
4: -400.4219360, 1179.3555908, -226.3007965, 657.4593506, -1057.8812256, 1405.6563721

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755807, upper bound: 743.6765352
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755807, upper bound: 743.6765352
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -213.4159546, 1144.1799316, -118.7505341, 626.8072510, -840.2232056, 1262.9304199
1: -351.9371643, 1357.1868896, -193.4060364, 742.4365234, -1094.3736572, 1550.5927734
2: -247.8705597, 1404.9827881, -137.3754883, 771.8521118, -1019.7226562, 1542.3581543
3: -608.2764282, 1188.0802002, -334.6438904, 646.7448730, -1255.0209961, 1522.7241211
4: -408.3177185, 1202.6004639, -226.3007965, 657.4593506, -1065.7769775, 1428.9011230

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755807, upper bound: 743.6765356
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755807, upper bound: 743.6765356
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -211.6472626, 1135.2235107, -118.0685501, 624.9121094, -836.5593872, 1253.2917480
1: -348.9617615, 1346.4598389, -192.6406555, 740.0949707, -1089.0565186, 1539.1003418
2: -245.8531952, 1393.9073486, -136.6780853, 769.4288940, -1015.2821045, 1530.5854492
3: -603.2770386, 1178.5952148, -333.1760864, 644.6849365, -1247.9619141, 1511.7708740
4: -405.0206909, 1192.9881592, -225.0942688, 655.1671753, -1060.1878662, 1418.0823975

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748768, upper bound: 743.6755767
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748768, upper bound: 743.6755767
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -215.8473663, 1155.7645264, -117.1353607, 613.9246826, -829.7719727, 1272.8999023
1: -355.3401489, 1371.6437988, -190.2110901, 728.4483032, -1083.7884521, 1561.8548584
2: -250.4403992, 1418.8759766, -135.3050385, 755.6836548, -1006.1240234, 1554.1810303
3: -614.1268311, 1202.1851807, -329.4906006, 636.8835449, -1251.0103760, 1531.6757812
4: -412.6138916, 1215.7668457, -223.4666290, 645.3305054, -1057.9443359, 1439.2332764

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708535, upper bound: 743.6697760
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754999, upper bound: 743.6755120
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -219.2081146, 1173.9383545, -117.1353607, 613.9246826, -833.1328125, 1291.0734863
1: -360.6501465, 1393.4038086, -190.2110901, 728.4483032, -1089.0982666, 1583.6148682
2: -254.3073730, 1441.0878906, -135.3050385, 755.6836548, -1009.9907837, 1576.3929443
3: -623.2453003, 1221.3461914, -329.4906006, 636.8835449, -1260.1286621, 1550.8367920
4: -419.1094971, 1234.7418213, -223.4666290, 645.3305054, -1064.4399414, 1458.2082520

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708535, upper bound: 743.6697760
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754999, upper bound: 743.6755120
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -216.1133423, 1157.0812988, -214.9396057, 1145.2023926, -1361.3156738, 1372.0208740
1: -355.7760315, 1373.2065430, -352.7152405, 1358.8963623, -1714.6719971, 1725.9215088
2: -250.7474823, 1420.5035400, -249.2037811, 1406.5161133, -1657.2634277, 1669.7072754
3: -614.8787231, 1203.5784912, -610.8955078, 1191.6590576, -1806.5374756, 1814.4739990
4: -413.1237183, 1217.1843262, -411.1761475, 1206.0214844, -1619.1451416, 1628.3603516

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -219.4756012, 1175.2653809, -214.9396057, 1145.2023926, -1364.6779785, 1390.2048340
1: -361.0865784, 1394.9780273, -352.7152405, 1358.8963623, -1719.9829102, 1747.6931152
2: -254.6149750, 1442.7270508, -249.2037811, 1406.5161133, -1661.1311035, 1691.9307861
3: -623.9979248, 1222.7460938, -610.8955078, 1191.6590576, -1815.6566162, 1833.6416016
4: -419.6197815, 1236.1679688, -411.1761475, 1206.0214844, -1625.6412354, 1647.3441162

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752212, upper bound: 743.6757584
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -215.8522949, 1155.7889404, -118.8719330, 622.3225098, -838.1747437, 1274.6606445
1: -355.3482056, 1371.6726074, -193.3012085, 738.4765015, -1093.8247070, 1564.9738770
2: -250.4460602, 1418.9060059, -137.3626099, 766.0497437, -1016.4957886, 1556.2685547
3: -614.1408081, 1202.2105713, -334.7113647, 645.7933350, -1259.9340820, 1536.9218750
4: -412.6231995, 1215.7934570, -226.7326508, 654.5278931, -1067.1510010, 1442.5257568

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6670676, upper bound: 743.6671520
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755070, upper bound: 743.6756010
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -219.2130890, 1173.9630127, -118.8719330, 622.3225098, -841.5355835, 1292.8349609
1: -360.6582642, 1393.4333496, -193.3012085, 738.4765015, -1099.1347656, 1586.7346191
2: -254.3130646, 1441.1181641, -137.3626099, 766.0497437, -1020.3626709, 1578.4807129
3: -623.2591553, 1221.3718262, -334.7113647, 645.7933350, -1269.0522461, 1556.0832520
4: -419.1189880, 1234.7681885, -226.7326508, 654.5278931, -1073.6468506, 1461.5006104

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6670676, upper bound: 743.6671702
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755070, upper bound: 743.6756010
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -216.1133423, 1157.0812988, -220.8299866, 1177.4466553, -1393.5595703, 1377.9112549
1: -355.7760315, 1373.2065430, -362.8426208, 1397.4820557, -1753.2579346, 1736.0491943
2: -250.7474823, 1420.5035400, -256.1611938, 1445.9011230, -1696.6485596, 1676.6647949
3: -614.8787231, 1203.5784912, -628.0755615, 1226.0454102, -1840.9239502, 1831.6539307
4: -413.1237183, 1217.1843262, -422.6713867, 1240.3104248, -1653.4339600, 1639.8555908

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -219.4756012, 1175.2653809, -220.8299866, 1177.4466553, -1396.9219971, 1396.0953369
1: -361.0865784, 1394.9780273, -362.8426208, 1397.4820557, -1758.5686035, 1757.8206787
2: -254.6149750, 1442.7270508, -256.1611938, 1445.9011230, -1700.5161133, 1698.8881836
3: -623.9979248, 1222.7460938, -628.0755615, 1226.0454102, -1850.0430908, 1850.8216553
4: -419.6197815, 1236.1679688, -422.6713867, 1240.3104248, -1659.9301758, 1658.8393555

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736741, upper bound: 743.6739234
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -204.4863129, 1095.8936768, -120.3839111, 631.0923462, -835.5786743, 1216.2774658
1: -337.0567017, 1300.3742676, -195.7545166, 748.8900146, -1085.9466553, 1496.1282959
2: -237.3348236, 1345.3878174, -139.2353973, 776.7808228, -1014.1156616, 1484.6230469
3: -582.4991455, 1139.3822021, -338.9459534, 655.0001221, -1237.4990234, 1478.3281250
4: -391.1440430, 1152.5740967, -229.9523315, 663.6384277, -1054.7822266, 1382.5263672

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757800, upper bound: 743.6766437
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757800, upper bound: 743.6778595
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -204.8859406, 1097.8942871, -219.9978027, 1173.0251465, -1377.9111328, 1317.8919678
1: -337.7149658, 1302.7408447, -361.2714539, 1392.2609863, -1729.9758301, 1664.0123291
2: -237.7970123, 1347.8659668, -255.2620087, 1440.5024414, -1678.2993164, 1603.1279297
3: -583.6344604, 1141.4816895, -625.5038452, 1221.6484375, -1805.2829590, 1766.9854736
4: -391.9103699, 1154.7204590, -421.3590393, 1235.8757324, -1627.7861328, 1576.0794678

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -209.7590485, 1124.2286377, -120.3839111, 631.0923462, -840.8513794, 1244.6121826
1: -345.5965271, 1334.0587158, -195.7545166, 748.8900146, -1094.4864502, 1529.8128662
2: -243.4864197, 1380.1778564, -139.2353973, 776.7808228, -1020.2672119, 1519.4130859
3: -597.1467896, 1168.8951416, -338.9459534, 655.0001221, -1252.1467285, 1507.8410645
4: -401.3127441, 1182.2614746, -229.9523315, 663.6384277, -1064.9511719, 1412.0382080

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757781, upper bound: 743.6755004
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757781, upper bound: 743.6778629
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -210.1590424, 1126.2252197, -219.9978027, 1173.0251465, -1383.1842041, 1346.2230225
1: -346.2530212, 1336.4221191, -361.2714539, 1392.2609863, -1738.5140381, 1697.6936035
2: -243.9489441, 1382.6511230, -255.2620087, 1440.5024414, -1684.4511719, 1637.9130859
3: -598.2797852, 1170.9915771, -625.5038452, 1221.6484375, -1819.9279785, 1796.4953613
4: -402.0786438, 1184.4053955, -421.3590393, 1235.8757324, -1637.9543457, 1605.7644043

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -207.9731293, 1115.4129639, -121.2836075, 635.9880371, -843.9611816, 1236.6961670
1: -342.6049500, 1323.5523682, -197.2377930, 754.7309570, -1097.3359375, 1520.7901611
2: -241.4162903, 1369.2910156, -140.2820740, 782.7495728, -1024.1658936, 1509.5731201
3: -592.1908569, 1159.6181641, -341.5011597, 660.1770630, -1252.3679199, 1501.1191406
4: -397.8311768, 1172.7574463, -231.6932373, 668.7517090, -1066.5826416, 1404.4504395

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770208, upper bound: 743.6766437
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770208, upper bound: 743.6781269
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -208.2442474, 1116.7576904, -221.1028442, 1179.4866943, -1387.7308350, 1337.8602295
1: -343.0488892, 1325.1466064, -363.1227417, 1399.9666748, -1743.0152588, 1688.2692871
2: -241.7290955, 1370.9554443, -256.5639038, 1448.3803711, -1690.1094971, 1627.5192871
3: -592.9567261, 1161.0368652, -628.6686401, 1228.3985596, -1821.3552246, 1789.7055664
4: -398.3496399, 1174.2047119, -423.5334167, 1242.5338135, -1640.8834229, 1597.7381592

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -212.9937897, 1142.0667725, -121.2836075, 635.9880371, -848.9818115, 1263.3499756
1: -350.6746826, 1355.3741455, -197.2377930, 754.7309570, -1105.4056396, 1552.6119385
2: -247.2205505, 1401.9732666, -140.2820740, 782.7495728, -1029.9700928, 1542.2553711
3: -606.0404663, 1187.6727295, -341.5011597, 660.1770630, -1266.2175293, 1529.1738281
4: -407.4759521, 1200.8050537, -231.6932373, 668.7517090, -1076.2275391, 1432.4981689

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6753346, upper bound: 743.6750516
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6753346, upper bound: 743.6762316
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -213.2642059, 1143.4080811, -221.1028442, 1179.4866943, -1392.7507324, 1364.5108643
1: -351.1159363, 1356.9638672, -363.1227417, 1399.9666748, -1751.0826416, 1720.0865479
2: -247.5319061, 1403.6311035, -256.5639038, 1448.3803711, -1695.9122314, 1660.1950684
3: -606.8019409, 1189.0869141, -628.6686401, 1228.3985596, -1835.2004395, 1817.7556152
4: -407.9923096, 1202.2469482, -423.5334167, 1242.5338135, -1650.5261230, 1625.7803955

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.55 + 342.39 = 344.94 seconds
