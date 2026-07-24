## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.055158916499999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479)
1: (-0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910)
2: (-0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850)
3: (-0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681)
4: (-0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295)

## BASE Result
execution time: IAR + LP analysis = 1.97 + 0.86 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0562259, upper bound: 0.0562259


# Binary Search by BASE starts (time budget: 1197.17 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=0.058847926557064056
rel_dist={0: [-0.05599888835187894, 0.05599888835187895]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=0.058847926557064056
rel_dist={0: [-0.05565336822157154, 0.055653368221571534]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=0.058847926557064056
rel_dist={0: [-0.05542113047992016, 0.055421130479920144]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=0.058847926557064056
rel_dist={0: [-0.05528714416572913, 0.05528714416572915]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=0.058847926557064056
rel_dist={0: [-0.055212698399098425, 0.055212698398983934]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.VERIFIED, low=0.0031250, high=0.0062500, mid=0.0031250, abs_max=0.058847926557064056
rel_dist={0: [-0.055132520784768234, 0.05513252078476821]}

## Binary search (step 6) starts
Candidate diff: 0.0046875


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0031250, high=0.0046875, mid=0.0046875, abs_max=0.058847926557064056
rel_dist={0: [-0.055185278748120487, 0.05518527874783165]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0031250, high=0.0039062, mid=0.0039062, abs_max=0.058847926557064056
rel_dist={0: [-0.055165241381868235, 0.05516524138172432]}

## Binary search (step 8) starts
Candidate diff: 0.0035156


## IAR start
Binary search (step 8): status=Status.VERIFIED, low=0.0035156, high=0.0039062, mid=0.0035156, abs_max=0.058847926557064056
rel_dist={0: [-0.05515501399134123, 0.055155013991219096]}

## Binary search (step 9) starts
Candidate diff: 0.0037109


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0035156, high=0.0037109, mid=0.0037109, abs_max=0.058847926557064056
rel_dist={0: [-0.05516014090469245, 0.055160140904564114]}

## Binary search (step 10) starts
Candidate diff: 0.0036133


## IAR start
Binary search (step 10): status=Status.VERIFIED, low=0.0036133, high=0.0037109, mid=0.0036133, abs_max=0.058847926557064056
rel_dist={0: [-0.05515757745100269, 0.05515757745087749]}

## Binary search (step 11) starts
Candidate diff: 0.0036621


## IAR start
Binary search (step 11): status=Status.VERIFIED, low=0.0036621, high=0.0037109, mid=0.0036621, abs_max=0.058847926557064056
rel_dist={0: [-0.05515885917960294, 0.05515885917960295]}

## Binary search (step 12) starts
Candidate diff: 0.0036865


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0036621, high=0.0036865, mid=0.0036865, abs_max=0.058847926557064056
rel_dist={0: [-0.05515950004240834, 0.05515950004228079]}

## Binary search (step 13) starts
Candidate diff: 0.0036743


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0036621, high=0.0036743, mid=0.0036743, abs_max=0.058847926557064056
rel_dist={0: [-0.0551591796108497, 0.05515917961084968]}

## Binary search (step 14) starts
Candidate diff: 0.0036682


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0036621, high=0.0036682, mid=0.0036682, abs_max=0.058847926557064056
rel_dist={0: [-0.05515901939457894, 0.05515901939432502]}

## Binary search (step 15) starts
Candidate diff: 0.0036652


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0036621, high=0.0036652, mid=0.0036652, abs_max=0.058847926557064056
rel_dist={0: [-0.05515893928599323, 0.05515893928586636]}

## Binary search (step 16) starts
Candidate diff: 0.0036636


## IAR start
Binary search (step 16): status=Status.VERIFIED, low=0.0036636, high=0.0036652, mid=0.0036636, abs_max=0.058847926557064056
rel_dist={0: [-0.05515889923307673, 0.05515889923294992]}

## Binary search (step 17) starts
Candidate diff: 0.0036644


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0036636, high=0.0036644, mid=0.0036644, abs_max=0.058847926557064056
rel_dist={0: [-0.05515891925993985, 0.05515891925968616]}

## Binary Search Result
Binary search time: 50.44 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.003663635035536572


# Individual Split (IS_dual) starts
Time budget: 1146.73 seconds

## Binary search (step 0) starts
Candidate diff: 0.1018318


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556653
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556653
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0278593, 0.0309887, -0.0516487, 0.0494817
1: -0.0226827, 0.0442280, -0.0350513, 0.0705397, -0.0932224, 0.0792793
2: -0.0535189, 0.0294451, -0.0677722, 0.0423128, -0.0958317, 0.0972173
3: -0.0368305, 0.0571968, -0.0527389, 0.0981292, -0.1349597, 0.1099357
4: -0.0685483, 0.0351860, -0.0944105, 0.0499190, -0.1184673, 0.1295964

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556578
time: 0.29 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556638
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0275995, 0.0305874, -0.0519514, 0.0514667
1: -0.0256137, 0.0480500, -0.0343877, 0.0693975, -0.0950112, 0.0824377
2: -0.0530172, 0.0317309, -0.0671558, 0.0413970, -0.0944141, 0.0988867
3: -0.0398649, 0.0629934, -0.0515191, 0.0963091, -0.1361740, 0.1145125
4: -0.0719933, 0.0355104, -0.0931684, 0.0484524, -0.1204456, 0.1286788

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.63 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556578
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556638
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0556482, upper bound: 0.0556482

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0278593, 0.0309887, -0.0484289, 0.0462473
1: -0.0179365, 0.0346897, -0.0350513, 0.0705397, -0.0884762, 0.0697410
2: -0.0468226, 0.0235523, -0.0677722, 0.0423128, -0.0891354, 0.0913245
3: -0.0312331, 0.0437208, -0.0527389, 0.0981292, -0.1293623, 0.0964598
4: -0.0597628, 0.0287614, -0.0944105, 0.0499190, -0.1096818, 0.1231719

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547504, upper bound: 0.0554792
time: 0.30 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556578
time: 0.32 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0278216, 0.0309204, -0.0587192, 0.0594372
1: -0.0329395, 0.0745442, -0.0349473, 0.0703340, -0.1032735, 0.1094915
2: -0.0723793, 0.0529296, -0.0676680, 0.0421710, -0.1145503, 0.1205976
3: -0.0475005, 0.0975785, -0.0525602, 0.0978113, -0.1453118, 0.1501387
4: -0.0997654, 0.0589750, -0.0941751, 0.0497064, -0.1494718, 0.1531501

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556270
time: 0.31 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556638
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0238139, 0.0252115, -0.0465755, 0.0476811
1: -0.0256137, 0.0480500, -0.0273273, 0.0526859, -0.0782996, 0.0753774
2: -0.0530172, 0.0317309, -0.0592544, 0.0342942, -0.0873114, 0.0909853
3: -0.0398649, 0.0629934, -0.0429685, 0.0698176, -0.1096825, 0.1059620
4: -0.0719933, 0.0355104, -0.0768934, 0.0406088, -0.1126021, 0.1124038

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
time: 0.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0213278, 0.0238178, -0.0435172, 0.0564903, -0.0778181, 0.0673350
1: -0.0255377, 0.0478916, -0.0788307, 0.1398573, -0.1653950, 0.1267222
2: -0.0529076, 0.0316422, -0.1017636, 0.0953863, -0.1482939, 0.1334058
3: -0.0397803, 0.0627622, -0.1304057, 0.2085697, -0.2483500, 0.1931680
4: -0.0718101, 0.0354106, -0.1687937, 0.1288396, -0.2006497, 0.2042043

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0552530
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0556482
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.66 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0547504, upper bound: 0.0554792
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556578
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556270
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0555700, upper bound: 0.0556638
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0552530
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -0.0555630, upper bound: 0.0556482

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0167172, 0.0175819, -0.0277201, 0.0323688, -0.0490861, 0.0453020
1: -0.0167369, 0.0320856, -0.0308648, 0.0743695, -0.0911064, 0.0629505
2: -0.0453822, 0.0222232, -0.0791640, 0.0529521, -0.0983343, 0.1013873
3: -0.0296628, 0.0403036, -0.0498447, 0.1098390, -0.1395018, 0.0901482
4: -0.0576971, 0.0273277, -0.1205781, 0.0659159, -0.1236130, 0.1479058

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546201, upper bound: 0.0554128
time: 0.28 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547131, upper bound: 0.0554180
time: 0.29 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0253249, 0.0272606, -0.0447009, 0.0437129
1: -0.0179365, 0.0346897, -0.0293032, 0.0589747, -0.0769111, 0.0639929
2: -0.0468226, 0.0235523, -0.0625771, 0.0371501, -0.0839727, 0.0861294
3: -0.0312331, 0.0437208, -0.0449141, 0.0797573, -0.1109904, 0.0886349
4: -0.0597628, 0.0287614, -0.0838840, 0.0437622, -0.1035250, 0.1126454

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556398
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556578
time: 0.30 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0206258, 0.0215764, -0.0493751, 0.0522413
1: -0.0329395, 0.0745442, -0.0226243, 0.0440746, -0.0770141, 0.0971685
2: -0.0723793, 0.0529296, -0.0534147, 0.0293607, -0.1017400, 0.1063443
3: -0.0475005, 0.0975785, -0.0367476, 0.0569787, -0.1044792, 0.1343261
4: -0.0997654, 0.0589750, -0.0683751, 0.0350883, -0.1348538, 0.1273501

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556258
time: 0.30 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0213278, 0.0238178, -0.0516166, 0.0529433
1: -0.0329395, 0.0745442, -0.0255377, 0.0478916, -0.0808311, 0.1000819
2: -0.0723793, 0.0529296, -0.0529076, 0.0316422, -0.1040216, 0.1058372
3: -0.0475005, 0.0975785, -0.0397803, 0.0627622, -0.1102628, 0.1373588
4: -0.0997654, 0.0589750, -0.0718101, 0.0354106, -0.1351760, 0.1307851

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556337
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0174403, 0.0183881, -0.0397520, 0.0413075
1: -0.0256137, 0.0480500, -0.0179365, 0.0346897, -0.0603034, 0.0659865
2: -0.0530172, 0.0317309, -0.0468226, 0.0235523, -0.0765695, 0.0785535
3: -0.0398649, 0.0629934, -0.0312331, 0.0437208, -0.0835857, 0.0942266
4: -0.0719933, 0.0355104, -0.0597628, 0.0287614, -0.1007547, 0.0952732

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552181, upper bound: 0.0547504
time: 0.31 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0193350, 0.0216215, -0.0429854, 0.0432022
1: -0.0256137, 0.0480500, -0.0221682, 0.0413110, -0.0669248, 0.0702183
2: -0.0530172, 0.0317309, -0.0484735, 0.0279975, -0.0810147, 0.0802044
3: -0.0398649, 0.0629934, -0.0360067, 0.0530757, -0.0929406, 0.0990001
4: -0.0719933, 0.0355104, -0.0654610, 0.0313325, -0.1033258, 0.1009714

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0551976
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0555630
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0435172, 0.0564903, -0.0758253, 0.0651387
1: -0.0221682, 0.0413110, -0.0788307, 0.1398573, -0.1620255, 0.1201417
2: -0.0484735, 0.0279975, -0.1017636, 0.0953863, -0.1438598, 0.1297611
3: -0.0360067, 0.0530757, -0.1304057, 0.2085697, -0.2445764, 0.1834815
4: -0.0654610, 0.0313325, -0.1687937, 0.1288396, -0.1943006, 0.2001262

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0435172, 0.0564903, -0.0814430, 0.0724614
1: -0.0302124, 0.0647534, -0.0788307, 0.1398573, -0.1700696, 0.1435841
2: -0.0640738, 0.0455330, -0.1017636, 0.0953863, -0.1594601, 0.1472966
3: -0.0429836, 0.0844141, -0.1304057, 0.2085697, -0.2515533, 0.2148198
4: -0.0914269, 0.0490248, -0.1687937, 0.1288396, -0.2202665, 0.2178185

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0555648
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0555648
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.72 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0546201, upper bound: 0.0554128
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0547131, upper bound: 0.0554180
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556398
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556578
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556258
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556254
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556337
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552181, upper bound: 0.0547504
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555630
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0551976
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0555630
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0555648
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0555648

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0276999, 0.0323285, -0.0505833, 0.0460298
1: -0.0175492, 0.0310600, -0.0308061, 0.0742758, -0.0918250, 0.0618661
2: -0.0452677, 0.0227551, -0.0791176, 0.0528919, -0.0981596, 0.1018728
3: -0.0302430, 0.0378163, -0.0497446, 0.1097040, -0.1399470, 0.0875608
4: -0.0534865, 0.0273936, -0.1204893, 0.0658278, -0.1193143, 0.1478828

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546143, upper bound: 0.0553128
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546201, upper bound: 0.0554127
time: 0.30 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0144043, 0.0128454, -0.0277201, 0.0323688, -0.0467731, 0.0405655
1: -0.0132008, 0.0249592, -0.0308648, 0.0743695, -0.0875703, 0.0558240
2: -0.0386996, 0.0166129, -0.0791640, 0.0529521, -0.0916518, 0.0957769
3: -0.0253240, 0.0303233, -0.0498447, 0.1098390, -0.1351630, 0.0801680
4: -0.0478803, 0.0207241, -0.1205781, 0.0659159, -0.1137962, 0.1413022

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546894, upper bound: 0.0551496
time: 0.29 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547131, upper bound: 0.0554180
time: 0.30 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0224162, 0.0235390, -0.0409793, 0.0408043
1: -0.0179365, 0.0346897, -0.0244602, 0.0471920, -0.0651284, 0.0591499
2: -0.0468226, 0.0235523, -0.0564170, 0.0319501, -0.0787727, 0.0799693
3: -0.0312331, 0.0437208, -0.0391877, 0.0615597, -0.0927929, 0.0829086
4: -0.0597628, 0.0287614, -0.0724862, 0.0380577, -0.0978205, 0.1012476

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556398
time: 0.31 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556398
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0421270, 0.0516748, -0.0691151, 0.0605150
1: -0.0179365, 0.0346897, -0.0601539, 0.1331329, -0.1510694, 0.0948436
2: -0.0468226, 0.0235523, -0.0991491, 0.0719263, -0.1187489, 0.1227014
3: -0.0312331, 0.0437208, -0.0832208, 0.1978890, -0.2291221, 0.1269417
4: -0.0597628, 0.0287614, -0.1639526, 0.0807114, -0.1404742, 0.1927140

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556400
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556578
time: 0.29 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0174403, 0.0183881, -0.0461869, 0.0490558
1: -0.0329395, 0.0745442, -0.0179365, 0.0346897, -0.0676292, 0.0924807
2: -0.0723793, 0.0529296, -0.0468226, 0.0235523, -0.0959316, 0.0997523
3: -0.0475005, 0.0975785, -0.0312331, 0.0437208, -0.0912214, 0.1288116
4: -0.0997654, 0.0589750, -0.0597628, 0.0287614, -0.1285269, 0.1187378

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0556167
time: 0.31 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553232, upper bound: 0.0553575
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0277988, 0.0316155, -0.0594143, 0.0594143
1: -0.0329395, 0.0745442, -0.0329395, 0.0745442, -0.1074837, 0.1074837
2: -0.0723793, 0.0529296, -0.0723793, 0.0529296, -0.1253090, 0.1253090
3: -0.0475005, 0.0975785, -0.0475005, 0.0975785, -0.1450790, 0.1450790
4: -0.0997654, 0.0589750, -0.0997654, 0.0589750, -0.1587404, 0.1587404

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551970, upper bound: 0.0535317
time: 0.28 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556146, upper bound: 0.0556146
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0193350, 0.0216215, -0.0494203, 0.0509505
1: -0.0329395, 0.0745442, -0.0221682, 0.0413110, -0.0742506, 0.0967125
2: -0.0723793, 0.0529296, -0.0484735, 0.0279975, -0.1003769, 0.1014031
3: -0.0475005, 0.0975785, -0.0360067, 0.0530757, -0.1005763, 0.1335852
4: -0.0997654, 0.0589750, -0.0654610, 0.0313325, -0.1310980, 0.1244360

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555993
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550420, upper bound: 0.0554502
time: 0.32 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0249527, 0.0289441, -0.0567429, 0.0565682
1: -0.0329395, 0.0745442, -0.0302124, 0.0647534, -0.0976930, 0.1047566
2: -0.0723793, 0.0529296, -0.0640738, 0.0455330, -0.1179123, 0.1170035
3: -0.0475005, 0.0975785, -0.0429836, 0.0844141, -0.1319146, 0.1405621
4: -0.0997654, 0.0589750, -0.0914269, 0.0490248, -0.1487902, 0.1504019

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0537097
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550383, upper bound: 0.0556163
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0200828, 0.0232392, -0.0167172, 0.0175819, -0.0376647, 0.0399565
1: -0.0189451, 0.0503686, -0.0167369, 0.0320856, -0.0510307, 0.0671055
2: -0.0627839, 0.0392658, -0.0453822, 0.0222232, -0.0850071, 0.0846480
3: -0.0322327, 0.0689562, -0.0296628, 0.0403036, -0.0725363, 0.0986190
4: -0.0872019, 0.0471414, -0.0576971, 0.0273277, -0.1145296, 0.1048386

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554128, upper bound: 0.0546201
time: 0.29 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554180, upper bound: 0.0547131
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0195409, 0.0220395, -0.0174403, 0.0183881, -0.0379290, 0.0394798
1: -0.0217663, 0.0421363, -0.0179365, 0.0346897, -0.0564560, 0.0600728
2: -0.0493362, 0.0288256, -0.0468226, 0.0235523, -0.0728885, 0.0756483
3: -0.0353122, 0.0542807, -0.0312331, 0.0437208, -0.0790330, 0.0855138
4: -0.0671461, 0.0321655, -0.0597628, 0.0287614, -0.0959075, 0.0919283

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556398, upper bound: 0.0551976
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556398, upper bound: 0.0555700
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0193350, 0.0216215, -0.0409564, 0.0409564
1: -0.0221682, 0.0413110, -0.0221682, 0.0413110, -0.0634793, 0.0634793
2: -0.0484735, 0.0279975, -0.0484735, 0.0279975, -0.0764710, 0.0764710
3: -0.0360067, 0.0530757, -0.0360067, 0.0530757, -0.0890824, 0.0890824
4: -0.0654610, 0.0313325, -0.0654610, 0.0313325, -0.0967935, 0.0967935

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 4.91 seconds

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552340, upper bound: 0.0551061
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551264, upper bound: 0.0551030
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0193350, 0.0216215, -0.0465741, 0.0482791
1: -0.0302124, 0.0647534, -0.0221682, 0.0413110, -0.0715234, 0.0869217
2: -0.0640738, 0.0455330, -0.0484735, 0.0279975, -0.0920714, 0.0940065
3: -0.0429836, 0.0844141, -0.0360067, 0.0530757, -0.0960593, 0.1204208
4: -0.0914269, 0.0490248, -0.0654610, 0.0313325, -0.1227595, 0.1144858

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.96 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536435, upper bound: 0.0548925
time: 0.31 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0553366
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0414968, 0.0538694, -0.0732044, 0.0631183
1: -0.0221682, 0.0413110, -0.0741096, 0.1330418, -0.1552100, 0.1154206
2: -0.0484735, 0.0279975, -0.0972371, 0.0890556, -0.1375291, 0.1252346
3: -0.0360067, 0.0530757, -0.1205643, 0.1977571, -0.2337638, 0.1736400
4: -0.0654610, 0.0313325, -0.1608542, 0.1174676, -0.1829286, 0.1921867

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0547001
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554441, upper bound: 0.0550420
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0237000, 0.0276457, -0.0469807, 0.0453215
1: -0.0221682, 0.0413110, -0.0270187, 0.0608168, -0.0829850, 0.0683298
2: -0.0484735, 0.0279975, -0.0619147, 0.0439293, -0.0924028, 0.0899122
3: -0.0360067, 0.0530757, -0.0392698, 0.0784101, -0.1144168, 0.0923455
4: -0.0654610, 0.0313325, -0.0880449, 0.0474173, -0.1128783, 0.1193774

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0547001
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554441, upper bound: 0.0550420
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0414968, 0.0538694, -0.0788221, 0.0704410
1: -0.0302124, 0.0647534, -0.0741096, 0.1330418, -0.1632541, 0.1388630
2: -0.0640738, 0.0455330, -0.0972371, 0.0890556, -0.1531294, 0.1427701
3: -0.0429836, 0.0844141, -0.1205643, 0.1977571, -0.2407407, 0.2049783
4: -0.0914269, 0.0490248, -0.1608542, 0.1174676, -0.2088945, 0.2098790

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552341, upper bound: 0.0536385
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556295, upper bound: 0.0555455
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0237000, 0.0276457, -0.0525984, 0.0526442
1: -0.0302124, 0.0647534, -0.0270187, 0.0608168, -0.0910291, 0.0917722
2: -0.0640738, 0.0455330, -0.0619147, 0.0439293, -0.1080031, 0.1074476
3: -0.0429836, 0.0844141, -0.0392698, 0.0784101, -0.1213937, 0.1236839
4: -0.0914269, 0.0490248, -0.0880449, 0.0474173, -0.1388442, 0.1370697

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537111, upper bound: 0.0551438
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556295, upper bound: 0.0555455
time: 0.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.62 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0546143, upper bound: 0.0553128
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0546201, upper bound: 0.0554127
IS_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0546894, upper bound: 0.0551496
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0547131, upper bound: 0.0554180
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556398
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556398
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556400
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556578
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0556167
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0553232, upper bound: 0.0553575
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0551970, upper bound: 0.0535317
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0556146, upper bound: 0.0556146
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555993
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0550420, upper bound: 0.0554502
IS_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0537097
IS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0550383, upper bound: 0.0556163
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0554128, upper bound: 0.0546201
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0554180, upper bound: 0.0547131
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0556398, upper bound: 0.0551976
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0556398, upper bound: 0.0555700
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0552340, upper bound: 0.0551061
IS_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0551264, upper bound: 0.0551030
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0536435, upper bound: 0.0548925
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0550479, upper bound: 0.0553366
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0547001
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0554441, upper bound: 0.0550420
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0547001
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0554441, upper bound: 0.0550420
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0552341, upper bound: 0.0536385
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0556295, upper bound: 0.0555455
IS_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0537111, upper bound: 0.0551438
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.62
Output dim: 0, lower bound: -0.0556295, upper bound: 0.0555455

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0264505, 0.0305609, -0.0488157, 0.0447804
1: -0.0175492, 0.0310600, -0.0270569, 0.0693507, -0.0868999, 0.0581169
2: -0.0452677, 0.0227551, -0.0762174, 0.0489095, -0.0941772, 0.0989725
3: -0.0302430, 0.0378163, -0.0429012, 0.1017318, -0.1319748, 0.0807175
4: -0.0534865, 0.0273936, -0.1151076, 0.0590241, -0.1125106, 0.1425011

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544041, upper bound: 0.0551218
time: 0.30 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536035, upper bound: 0.0550543
time: 0.29 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546110, upper bound: 0.0552915
time: 0.30 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0272284, 0.0314817, -0.0497366, 0.0455583
1: -0.0175492, 0.0310600, -0.0295614, 0.0723613, -0.0899105, 0.0606214
2: -0.0452677, 0.0227551, -0.0780314, 0.0512423, -0.0965100, 0.1007865
3: -0.0302430, 0.0378163, -0.0475212, 0.1068369, -0.1370799, 0.0853375
4: -0.0534865, 0.0273936, -0.1185196, 0.0633134, -0.1167999, 0.1459131

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544090, upper bound: 0.0552140
time: 0.31 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544054, upper bound: 0.0552355
time: 0.29 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0144043, 0.0128454, -0.0272482, 0.0315258, -0.0459300, 0.0400937
1: -0.0132008, 0.0249592, -0.0296192, 0.0724507, -0.0856516, 0.0545784
2: -0.0386996, 0.0166129, -0.0780765, 0.0513000, -0.0899997, 0.0946894
3: -0.0253240, 0.0303233, -0.0476144, 0.1069658, -0.1322898, 0.0779377
4: -0.0478803, 0.0207241, -0.1186049, 0.0633985, -0.1112787, 0.1393291

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544890, upper bound: 0.0551941
time: 0.30 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545095, upper bound: 0.0552476
time: 0.30 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0168666, 0.0177314, -0.0351716, 0.0352547
1: -0.0179365, 0.0346897, -0.0166844, 0.0324948, -0.0504313, 0.0513741
2: -0.0468226, 0.0235523, -0.0457053, 0.0225558, -0.0693785, 0.0692576
3: -0.0312331, 0.0437208, -0.0294895, 0.0407429, -0.0719760, 0.0732103
4: -0.0597628, 0.0287614, -0.0583189, 0.0276646, -0.0874274, 0.0870804

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550358, upper bound: 0.0556297
time: 0.31 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0553991
time: 0.29 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0177528, 0.0200240, -0.0374643, 0.0361409
1: -0.0179365, 0.0346897, -0.0189958, 0.0363287, -0.0542652, 0.0536855
2: -0.0468226, 0.0235523, -0.0453282, 0.0254624, -0.0722850, 0.0688805
3: -0.0312331, 0.0437208, -0.0320662, 0.0460069, -0.0772400, 0.0757870
4: -0.0597628, 0.0287614, -0.0614046, 0.0284768, -0.0882396, 0.0901660

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548350, upper bound: 0.0536886
time: 0.30 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551589, upper bound: 0.0556282
time: 0.31 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0397151, 0.0488666, -0.0663069, 0.0581032
1: -0.0179365, 0.0346897, -0.0569106, 0.1248108, -0.1427473, 0.0916003
2: -0.0468226, 0.0235523, -0.0939133, 0.0682135, -0.1150361, 0.1174656
3: -0.0312331, 0.0437208, -0.0793627, 0.1845545, -0.2157876, 0.1230835
4: -0.0597628, 0.0287614, -0.1542809, 0.0764101, -0.1361729, 0.1830423

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555239, upper bound: 0.0549814
time: 0.31 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554067, upper bound: 0.0553232
time: 0.30 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0232206, 0.0272127, -0.0446530, 0.0416087
1: -0.0179365, 0.0346897, -0.0260053, 0.0594450, -0.0773815, 0.0606950
2: -0.0468226, 0.0235523, -0.0611328, 0.0432778, -0.0901004, 0.0846851
3: -0.0312331, 0.0437208, -0.0381173, 0.0764618, -0.1076949, 0.0818381
4: -0.0597628, 0.0287614, -0.0869171, 0.0466825, -0.1064453, 0.1156785

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555239, upper bound: 0.0549814
time: 0.32 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554067, upper bound: 0.0554748
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0174305, 0.0183766, -0.0462218, 0.0485283
1: -0.0329159, 0.0712517, -0.0179142, 0.0346495, -0.0675655, 0.0891659
2: -0.0701464, 0.0509891, -0.0468024, 0.0235350, -0.0936814, 0.0977915
3: -0.0480456, 0.0928499, -0.0312014, 0.0436644, -0.0917100, 0.1240513
4: -0.0942536, 0.0571052, -0.0597349, 0.0287415, -0.1229951, 0.1168401

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0553164
time: 0.30 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554067
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0174403, 0.0183881, -0.0423899, 0.0447174
1: -0.0260409, 0.0594983, -0.0179365, 0.0346897, -0.0607306, 0.0774348
2: -0.0639335, 0.0455851, -0.0468226, 0.0235523, -0.0874858, 0.0924077
3: -0.0402265, 0.0769224, -0.0312331, 0.0437208, -0.0839474, 0.1081556
4: -0.0868411, 0.0507081, -0.0597628, 0.0287614, -0.1156025, 0.1104710

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553232, upper bound: 0.0553164
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553232, upper bound: 0.0554067
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0199191, 0.0190619, -0.0276403, 0.0313810, -0.0513001, 0.0467022
1: -0.0243385, 0.0377710, -0.0327357, 0.0738348, -0.0981733, 0.0705067
2: -0.0425945, 0.0226177, -0.0718559, 0.0523944, -0.0949890, 0.0944736
3: -0.0361759, 0.0458784, -0.0472410, 0.0965818, -0.1327577, 0.0931194
4: -0.0496890, 0.0251871, -0.0988984, 0.0583906, -0.1080795, 0.1240855

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551721, upper bound: 0.0532256
time: 0.31 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550408, upper bound: 0.0535114
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0277988, 0.0316155, -0.0589182, 0.0588311
1: -0.0318687, 0.0727345, -0.0329395, 0.0745442, -0.1064129, 0.1056740
2: -0.0712754, 0.0519350, -0.0723793, 0.0529296, -0.1242050, 0.1243143
3: -0.0462107, 0.0947695, -0.0475005, 0.0975785, -0.1437892, 0.1422701
4: -0.0982611, 0.0578686, -0.0997654, 0.0589750, -0.1572361, 0.1576340

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556045, upper bound: 0.0550682
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0553158
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0193233, 0.0216105, -0.0494557, 0.0504211
1: -0.0329159, 0.0712517, -0.0221435, 0.0412708, -0.0741867, 0.0933952
2: -0.0701464, 0.0509891, -0.0484471, 0.0279795, -0.0981259, 0.0994362
3: -0.0480456, 0.0928499, -0.0359738, 0.0530184, -0.1010641, 0.1288237
4: -0.0942536, 0.0571052, -0.0654308, 0.0313087, -0.1255623, 0.1225360

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546678, upper bound: 0.0555964
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555517
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0193350, 0.0216215, -0.0456234, 0.0466121
1: -0.0260409, 0.0594983, -0.0221682, 0.0413110, -0.0673520, 0.0816665
2: -0.0639335, 0.0455851, -0.0484735, 0.0279975, -0.0919310, 0.0940586
3: -0.0402265, 0.0769224, -0.0360067, 0.0530757, -0.0933023, 0.1129292
4: -0.0868411, 0.0507081, -0.0654610, 0.0313325, -0.1181736, 0.1161691

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547900, upper bound: 0.0536888
time: 0.30 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549929, upper bound: 0.0554171
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0249527, 0.0289441, -0.0562468, 0.0559850
1: -0.0318687, 0.0727345, -0.0302124, 0.0647534, -0.0966221, 0.1029468
2: -0.0712754, 0.0519350, -0.0640738, 0.0455330, -0.1168084, 0.1160088
3: -0.0462107, 0.0947695, -0.0429836, 0.0844141, -0.1306248, 0.1377531
4: -0.0982611, 0.0578686, -0.0914269, 0.0490248, -0.1472859, 0.1492955

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551178, upper bound: 0.0549915
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0554455
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0200735, 0.0232269, -0.0182549, 0.0183299, -0.0384035, 0.0414818
1: -0.0189226, 0.0503497, -0.0175492, 0.0310600, -0.0499826, 0.0678989
2: -0.0627614, 0.0392500, -0.0452677, 0.0227551, -0.0855165, 0.0845178
3: -0.0322012, 0.0689265, -0.0302430, 0.0378163, -0.0700174, 0.0991695
4: -0.0871733, 0.0471209, -0.0534865, 0.0273936, -0.1145668, 0.1006074

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553128, upper bound: 0.0546143
time: 0.31 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554127, upper bound: 0.0546201
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0200828, 0.0232392, -0.0144043, 0.0128454, -0.0329282, 0.0376435
1: -0.0189451, 0.0503686, -0.0132008, 0.0249592, -0.0439043, 0.0635695
2: -0.0627839, 0.0392658, -0.0386996, 0.0166129, -0.0793968, 0.0779654
3: -0.0322327, 0.0689562, -0.0253240, 0.0303233, -0.0625560, 0.0942802
4: -0.0872019, 0.0471414, -0.0478803, 0.0207241, -0.1079260, 0.0950217

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0546894
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554180, upper bound: 0.0547131
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0177528, 0.0200240, -0.0174403, 0.0183881, -0.0361409, 0.0374643
1: -0.0189958, 0.0363287, -0.0179365, 0.0346897, -0.0536855, 0.0542652
2: -0.0453282, 0.0254624, -0.0468226, 0.0235523, -0.0688805, 0.0722850
3: -0.0320662, 0.0460069, -0.0312331, 0.0437208, -0.0757870, 0.0772400
4: -0.0614046, 0.0284768, -0.0597628, 0.0287614, -0.0901660, 0.0882396

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536886, upper bound: 0.0548350
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556282, upper bound: 0.0551589
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0244671, 0.0285040, -0.0174403, 0.0183881, -0.0428552, 0.0459443
1: -0.0291369, 0.0633202, -0.0179365, 0.0346897, -0.0638266, 0.0812567
2: -0.0632677, 0.0448866, -0.0468226, 0.0235523, -0.0868200, 0.0917093
3: -0.0417759, 0.0822587, -0.0312331, 0.0437208, -0.0854968, 0.1134919
4: -0.0902513, 0.0482991, -0.0597628, 0.0287614, -0.1190127, 0.1080619

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556188, upper bound: 0.0551293
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556048, upper bound: 0.0555192
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0171319, 0.0193760, -0.0193350, 0.0216215, -0.0387534, 0.0387109
1: -0.0185982, 0.0350459, -0.0221682, 0.0413110, -0.0599092, 0.0572142
2: -0.0431861, 0.0235984, -0.0484735, 0.0279975, -0.0711836, 0.0720719
3: -0.0321350, 0.0433074, -0.0360067, 0.0530757, -0.0852107, 0.0793141
4: -0.0576155, 0.0261779, -0.0654610, 0.0313325, -0.0889480, 0.0916389

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546749, upper bound: 0.0534765
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550313, upper bound: 0.0548674
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0163063, 0.0177085, -0.0426611, 0.0452504
1: -0.0302124, 0.0647534, -0.0184658, 0.0330927, -0.0633050, 0.0832193
2: -0.0640738, 0.0455330, -0.0422703, 0.0221831, -0.0862569, 0.0878033
3: -0.0429836, 0.0844141, -0.0314417, 0.0417711, -0.0847547, 0.1158558
4: -0.0914269, 0.0490248, -0.0568412, 0.0248886, -0.1163155, 0.1058660

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550313, upper bound: 0.0548965
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550082, upper bound: 0.0553366
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0193233, 0.0216105, -0.0393321, 0.0495266, -0.0688500, 0.0609426
1: -0.0221435, 0.0412708, -0.0687150, 0.1214286, -0.1435722, 0.1099858
2: -0.0484471, 0.0279795, -0.0912258, 0.0822822, -0.1307293, 0.1192053
3: -0.0359738, 0.0530184, -0.1112330, 0.1786075, -0.2145813, 0.1642515
4: -0.0654308, 0.0313087, -0.1461978, 0.1076009, -0.1730317, 0.1775064

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555964, upper bound: 0.0546678
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547768
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0362481, 0.0464668, -0.0658018, 0.0578695
1: -0.0221682, 0.0413110, -0.0602394, 0.1100473, -0.1322156, 0.1015504
2: -0.0484735, 0.0279975, -0.0875491, 0.0741147, -0.1225882, 0.1155466
3: -0.0360067, 0.0530757, -0.0969803, 0.1631809, -0.1991876, 0.1500560
4: -0.0654610, 0.0313325, -0.1404332, 0.0949922, -0.1604532, 0.1717658

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536888, upper bound: 0.0547900
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0549929
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0193233, 0.0216105, -0.0252150, 0.0284831, -0.0478064, 0.0468256
1: -0.0221435, 0.0412708, -0.0279653, 0.0610138, -0.0831573, 0.0692361
2: -0.0484471, 0.0279795, -0.0635368, 0.0439139, -0.0923610, 0.0915163
3: -0.0359738, 0.0530184, -0.0413184, 0.0779307, -0.1139044, 0.0943368
4: -0.0654308, 0.0313087, -0.0858417, 0.0480413, -0.1134721, 0.1171504

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546650
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0547001
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0198437, 0.0232866, -0.0426215, 0.0414652
1: -0.0221682, 0.0413110, -0.0199278, 0.0458330, -0.0680012, 0.0612388
2: -0.0484735, 0.0279975, -0.0528894, 0.0365279, -0.0850015, 0.0808869
3: -0.0360067, 0.0530757, -0.0318742, 0.0581797, -0.0941864, 0.0849499
4: -0.0654610, 0.0313325, -0.0742503, 0.0389823, -0.1044433, 0.1055828

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537198, upper bound: 0.0547859
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554135, upper bound: 0.0549929
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0222972, 0.0209378, -0.0413347, 0.0535006, -0.0757978, 0.0622725
1: -0.0313137, 0.0422930, -0.0737011, 0.1320493, -0.1633630, 0.1159941
2: -0.0440553, 0.0230310, -0.0967166, 0.0882339, -0.1322892, 0.1197476
3: -0.0458820, 0.0541391, -0.1197741, 0.1962066, -0.2420887, 0.1739132
4: -0.0546753, 0.0241777, -0.1596566, 0.1163448, -0.1710201, 0.1838343

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552066, upper bound: 0.0536385
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552341, upper bound: 0.0536241
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0243037, 0.0282464, -0.0414968, 0.0538694, -0.0781731, 0.0697433
1: -0.0291733, 0.0628643, -0.0741096, 0.1330418, -0.1622151, 0.1369739
2: -0.0628437, 0.0443860, -0.0972371, 0.0890556, -0.1518993, 0.1416230
3: -0.0415990, 0.0814768, -0.1205643, 0.1977571, -0.2393561, 0.2020411
4: -0.0897038, 0.0477546, -0.1608542, 0.1174676, -0.2071714, 0.2086088

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0555078
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0553158
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0230395, 0.0269408, -0.0518934, 0.0519836
1: -0.0302124, 0.0647534, -0.0258892, 0.0589280, -0.0891404, 0.0906427
2: -0.0640738, 0.0455330, -0.0607148, 0.0427832, -0.1068570, 0.1062478
3: -0.0429836, 0.0844141, -0.0377389, 0.0756101, -0.1185937, 0.1221530
4: -0.0914269, 0.0490248, -0.0863544, 0.0461579, -0.1375849, 0.1353792

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552725, upper bound: 0.0547058
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555927, upper bound: 0.0555155
time: 0.35 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.79 seconds
IS_A1_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0536035, upper bound: 0.0550543
IS_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0546110, upper bound: 0.0552915
IS_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0544090, upper bound: 0.0552140
IS_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0544054, upper bound: 0.0552355
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0544890, upper bound: 0.0551941
IS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0545095, upper bound: 0.0552476
IS_A1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550358, upper bound: 0.0556297
IS_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0553991
IS_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0548350, upper bound: 0.0536886
IS_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0551589, upper bound: 0.0556282
IS_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0555239, upper bound: 0.0549814
IS_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554067, upper bound: 0.0553232
IS_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0555239, upper bound: 0.0549814
IS_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554067, upper bound: 0.0554748
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0553164
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554067
IS_A1_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0553232, upper bound: 0.0553164
IS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0553232, upper bound: 0.0554067
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0551721, upper bound: 0.0532256
IS_A1_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550408, upper bound: 0.0535114
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0556045, upper bound: 0.0550682
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0553158
IS_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0546678, upper bound: 0.0555964
IS_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555517
IS_A1_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0547900, upper bound: 0.0536888
IS_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0549929, upper bound: 0.0554171
IS_A1_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0551178, upper bound: 0.0549915
IS_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0554455
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0553128, upper bound: 0.0546143
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554127, upper bound: 0.0546201
IS_A2_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0546894
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554180, upper bound: 0.0547131
IS_A2_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0536886, upper bound: 0.0548350
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0556282, upper bound: 0.0551589
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0556188, upper bound: 0.0551293
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0556048, upper bound: 0.0555192
IS_A2_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0546749, upper bound: 0.0534765
IS_A2_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550313, upper bound: 0.0548674
IS_A2_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550313, upper bound: 0.0548965
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0550082, upper bound: 0.0553366
IS_A2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0555964, upper bound: 0.0546678
IS_A2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547768
IS_A2_B2_A1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0536888, upper bound: 0.0547900
IS_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0549929
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546650
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0547001
IS_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0537198, upper bound: 0.0547859
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554135, upper bound: 0.0549929
IS_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0552066, upper bound: 0.0536385
IS_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0552341, upper bound: 0.0536241
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0555078
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0553158
IS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0552725, upper bound: 0.0547058
IS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.79
Output dim: 0, lower bound: -0.0555927, upper bound: 0.0555155

## BFS IS instance: IS_A1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0259758, 0.0298574, -0.0481122, 0.0443057
1: -0.0175492, 0.0310600, -0.0261192, 0.0674827, -0.0850320, 0.0571792
2: -0.0452677, 0.0227551, -0.0751136, 0.0479624, -0.0932301, 0.0978687
3: -0.0302430, 0.0378163, -0.0416708, 0.0986317, -0.1288747, 0.0794870
4: -0.0534865, 0.0273936, -0.1130735, 0.0579246, -0.1114111, 0.1404670

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544035, upper bound: 0.0551025
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544190, upper bound: 0.0551188
time: 0.31 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545923, upper bound: 0.0552520
time: 0.32 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0167531, 0.0159926, -0.0172754, 0.0129451, -0.0296982, 0.0332680
1: -0.0151484, 0.0245952, -0.0157733, 0.0281050, -0.0432534, 0.0403686
2: -0.0405012, 0.0175003, -0.0458170, 0.0141173, -0.0546185, 0.0633173
3: -0.0271696, 0.0289076, -0.0285681, 0.0368245, -0.0639941, 0.0574757
4: -0.0452045, 0.0217376, -0.0551897, 0.0220264, -0.0672309, 0.0769274

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542157, upper bound: 0.0549543
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543891, upper bound: 0.0551779
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0251972, 0.0266188, -0.0448737, 0.0435271
1: -0.0175492, 0.0310600, -0.0250645, 0.0642894, -0.0818386, 0.0561244
2: -0.0452677, 0.0227551, -0.0732274, 0.0455609, -0.0908286, 0.0959825
3: -0.0302430, 0.0378163, -0.0405258, 0.0936018, -0.1238447, 0.0783421
4: -0.0534865, 0.0273936, -0.1091430, 0.0556907, -0.1091772, 0.1365366

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532285, upper bound: 0.0548780
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544007, upper bound: 0.0552110
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0131821, 0.0112233, -0.0172916, 0.0129608, -0.0261429, 0.0285149
1: -0.0121467, 0.0204804, -0.0158059, 0.0281387, -0.0402855, 0.0362862
2: -0.0343113, 0.0120328, -0.0458563, 0.0141507, -0.0484620, 0.0578891
3: -0.0241067, 0.0235954, -0.0285961, 0.0368776, -0.0609844, 0.0521915
4: -0.0409769, 0.0154860, -0.0552379, 0.0220672, -0.0630441, 0.0707239

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540543, upper bound: 0.0546370
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540543, upper bound: 0.0544372
time: 0.36 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0144043, 0.0128454, -0.0252065, 0.0266334, -0.0410377, 0.0380520
1: -0.0132008, 0.0249592, -0.0250885, 0.0643248, -0.0775256, 0.0500477
2: -0.0386996, 0.0166129, -0.0732523, 0.0455832, -0.0842828, 0.0898652
3: -0.0253240, 0.0303233, -0.0405602, 0.0936605, -0.1189845, 0.0708835
4: -0.0478803, 0.0207241, -0.1091928, 0.0557172, -0.1035974, 0.1299169

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533375, upper bound: 0.0549366
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544968, upper bound: 0.0552202
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188422, 0.0191683, -0.0168594, 0.0177231, -0.0365652, 0.0360277
1: -0.0187277, 0.0338746, -0.0166682, 0.0324657, -0.0511934, 0.0505428
2: -0.0467068, 0.0241366, -0.0456893, 0.0225428, -0.0692496, 0.0698259
3: -0.0318076, 0.0415401, -0.0294663, 0.0407033, -0.0725109, 0.0710064
4: -0.0558105, 0.0288630, -0.0582974, 0.0276489, -0.0834594, 0.0871605

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552973, upper bound: 0.0552973
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552973, upper bound: 0.0553991
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0148684, 0.0150387, -0.0168666, 0.0177314, -0.0325998, 0.0319053
1: -0.0138386, 0.0261036, -0.0166844, 0.0324948, -0.0463334, 0.0427880
2: -0.0399446, 0.0177537, -0.0457053, 0.0225558, -0.0625004, 0.0634590
3: -0.0261081, 0.0320604, -0.0294895, 0.0407429, -0.0668510, 0.0615499
4: -0.0496877, 0.0220269, -0.0583189, 0.0276646, -0.0773523, 0.0803459

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0552973
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0553991
time: 0.35 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0167507, 0.0175735, -0.0177528, 0.0200240, -0.0367747, 0.0353263
1: -0.0169593, 0.0324132, -0.0189958, 0.0363287, -0.0532879, 0.0514090
2: -0.0453734, 0.0220414, -0.0453282, 0.0254624, -0.0708357, 0.0673696
3: -0.0299786, 0.0405239, -0.0320662, 0.0460069, -0.0759855, 0.0725901
4: -0.0578682, 0.0270928, -0.0614046, 0.0284768, -0.0863450, 0.0884974

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550043, upper bound: 0.0556163
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551134, upper bound: 0.0554171
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0174305, 0.0183766, -0.0377079, 0.0451381, -0.0625686, 0.0560845
1: -0.0179142, 0.0346495, -0.0533660, 0.1142647, -0.1321789, 0.0880155
2: -0.0468024, 0.0235350, -0.0882274, 0.0638561, -0.1106584, 0.1117624
3: -0.0312014, 0.0436644, -0.0744301, 0.1670350, -0.1982364, 0.1180945
4: -0.0597349, 0.0287415, -0.1403323, 0.0711920, -0.1309268, 0.1690738

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551168, upper bound: 0.0531659
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556045, upper bound: 0.0550343
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0346156, 0.0423562, -0.0597964, 0.0530037
1: -0.0179365, 0.0346897, -0.0472905, 0.1025978, -0.1205343, 0.0819802
2: -0.0468226, 0.0235523, -0.0845582, 0.0593169, -0.1061396, 0.1081105
3: -0.0312331, 0.0437208, -0.0671641, 0.1508321, -0.1820652, 0.1108849
4: -0.0597628, 0.0287614, -0.1345043, 0.0665109, -0.1262738, 0.1632657

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553164, upper bound: 0.0553232
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553164, upper bound: 0.0553232
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0174305, 0.0183766, -0.0247595, 0.0280647, -0.0454952, 0.0431361
1: -0.0179142, 0.0346495, -0.0269566, 0.0594490, -0.0773632, 0.0616061
2: -0.0468024, 0.0235350, -0.0626380, 0.0433056, -0.0901080, 0.0861730
3: -0.0312014, 0.0436644, -0.0402126, 0.0754785, -0.1066799, 0.0838770
4: -0.0597349, 0.0287415, -0.0846622, 0.0473183, -0.1070531, 0.1134037

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0545035
time: 0.38 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554596, upper bound: 0.0549814
time: 0.36 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0193921, 0.0228804, -0.0403206, 0.0377802
1: -0.0179365, 0.0346897, -0.0188992, 0.0442713, -0.0622078, 0.0535889
2: -0.0468226, 0.0235523, -0.0520622, 0.0359285, -0.0827512, 0.0756145
3: -0.0312331, 0.0437208, -0.0305969, 0.0559185, -0.0871516, 0.0743177
4: -0.0597628, 0.0287614, -0.0731249, 0.0382553, -0.0980181, 0.1018863

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545163, upper bound: 0.0551352
time: 0.39 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552732, upper bound: 0.0553590
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0188422, 0.0191683, -0.0470135, 0.0499399
1: -0.0329159, 0.0712517, -0.0187277, 0.0338746, -0.0667906, 0.0899794
2: -0.0701464, 0.0509891, -0.0467068, 0.0241366, -0.0942830, 0.0976960
3: -0.0480456, 0.0928499, -0.0318076, 0.0415401, -0.0895857, 0.1246575
4: -0.0942536, 0.0571052, -0.0558105, 0.0288630, -0.1231166, 0.1129157

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554519
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0148684, 0.0150387, -0.0428839, 0.0459661
1: -0.0329159, 0.0712517, -0.0138386, 0.0261036, -0.0590196, 0.0850903
2: -0.0701464, 0.0509891, -0.0399446, 0.0177537, -0.0879001, 0.0909337
3: -0.0480456, 0.0928499, -0.0261081, 0.0320604, -0.0801060, 0.1189580
4: -0.0942536, 0.0571052, -0.0496877, 0.0220269, -0.1162805, 0.1067929

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
time: 0.35 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555301
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0188422, 0.0191683, -0.0431702, 0.0461193
1: -0.0260409, 0.0594983, -0.0187277, 0.0338746, -0.0599155, 0.0782260
2: -0.0639335, 0.0455851, -0.0467068, 0.0241366, -0.0880701, 0.0922919
3: -0.0402265, 0.0769224, -0.0318076, 0.0415401, -0.0817666, 0.1087300
4: -0.0868411, 0.0507081, -0.0558105, 0.0288630, -0.1157041, 0.1065186

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550389, upper bound: 0.0535228
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552922
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0148684, 0.0150387, -0.0390405, 0.0421456
1: -0.0260409, 0.0594983, -0.0138386, 0.0261036, -0.0521445, 0.0733368
2: -0.0639335, 0.0455851, -0.0399446, 0.0177537, -0.0816872, 0.0855297
3: -0.0402265, 0.0769224, -0.0261081, 0.0320604, -0.0722870, 0.1030305
4: -0.0868411, 0.0507081, -0.0496877, 0.0220269, -0.1088680, 0.1003958

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550389, upper bound: 0.0536319
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552922
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0199149, 0.0190586, -0.0277199, 0.0308662, -0.0507810, 0.0467785
1: -0.0243291, 0.0377560, -0.0327256, 0.0705398, -0.0948688, 0.0704817
2: -0.0425885, 0.0226133, -0.0696913, 0.0504549, -0.0930434, 0.0923047
3: -0.0361665, 0.0458543, -0.0478097, 0.0918410, -0.1280075, 0.0936640
4: -0.0496803, 0.0251822, -0.0933773, 0.0565383, -0.1062186, 0.1185595

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551690, upper bound: 0.0530699
time: 0.31 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551440, upper bound: 0.0532256
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0272971, 0.0310283, -0.0278452, 0.0310977, -0.0583948, 0.0588735
1: -0.0318556, 0.0727205, -0.0329159, 0.0712517, -0.1031073, 0.1056364
2: -0.0712675, 0.0519287, -0.0701464, 0.0509891, -0.1222566, 0.1220750
3: -0.0461972, 0.0947469, -0.0480456, 0.0928499, -0.1390471, 0.1427925
4: -0.0982494, 0.0578616, -0.0942536, 0.0571052, -0.1553546, 0.1521152

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0240019, 0.0272771, -0.0545798, 0.0550342
1: -0.0318687, 0.0727345, -0.0260409, 0.0594983, -0.0913670, 0.0987754
2: -0.0712754, 0.0519350, -0.0639335, 0.0455851, -0.1168605, 0.1158685
3: -0.0462107, 0.0947695, -0.0402265, 0.0769224, -0.1231332, 0.1349961
4: -0.0982611, 0.0578686, -0.0868411, 0.0507081, -0.1489692, 0.1447096

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534948, upper bound: 0.0550412
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0534948, upper bound: 0.0553158
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0278679, 0.0321473, -0.0193233, 0.0216105, -0.0494784, 0.0514706
1: -0.0329683, 0.0750322, -0.0221435, 0.0412708, -0.0742391, 0.0971757
2: -0.0702526, 0.0525257, -0.0484471, 0.0279795, -0.0982321, 0.1009728
3: -0.0474031, 0.0969662, -0.0359738, 0.0530184, -0.1004216, 0.1329400
4: -0.0962531, 0.0578964, -0.0654308, 0.0313087, -0.1275617, 0.1233271

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.50 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0518935, upper bound: 0.0543927
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553700
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0273508, 0.0306702, -0.0193233, 0.0216105, -0.0489614, 0.0499935
1: -0.0321755, 0.0699852, -0.0221435, 0.0412708, -0.0734463, 0.0921288
2: -0.0691589, 0.0502736, -0.0484471, 0.0279795, -0.0971384, 0.0987207
3: -0.0471808, 0.0909255, -0.0359738, 0.0530184, -0.1001992, 0.1268993
4: -0.0932161, 0.0562352, -0.0654308, 0.0313087, -0.1245248, 0.1216660

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.61 seconds

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547440, upper bound: 0.0555517
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547734, upper bound: 0.0555517
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0193350, 0.0216215, -0.0451088, 0.0460196
1: -0.0250003, 0.0577051, -0.0221682, 0.0413110, -0.0663114, 0.0798734
2: -0.0628079, 0.0445847, -0.0484735, 0.0279975, -0.0908054, 0.0930582
3: -0.0388779, 0.0740361, -0.0360067, 0.0530757, -0.0919536, 0.1100428
4: -0.0853493, 0.0495821, -0.0654610, 0.0313325, -0.1166818, 0.1150431

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546241, upper bound: 0.0552869
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0554171
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0204420, 0.0239954, -0.0512981, 0.0514743
1: -0.0318687, 0.0727345, -0.0214377, 0.0480555, -0.0799242, 0.0941722
2: -0.0712754, 0.0519350, -0.0541373, 0.0375680, -0.1088433, 0.1060724
3: -0.0462107, 0.0947695, -0.0337864, 0.0613243, -0.1075350, 0.1285560
4: -0.0982611, 0.0578686, -0.0761260, 0.0400765, -0.1383376, 0.1339945

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554455
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554455
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0189814, 0.0215952, -0.0182549, 0.0183299, -0.0373113, 0.0398501
1: -0.0170536, 0.0488334, -0.0175492, 0.0310600, -0.0481136, 0.0663826
2: -0.0604146, 0.0375491, -0.0452677, 0.0227551, -0.0831697, 0.0828168
3: -0.0296645, 0.0664331, -0.0302430, 0.0378163, -0.0674808, 0.0966761
4: -0.0845780, 0.0446710, -0.0534865, 0.0273936, -0.1119716, 0.0981575

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551218, upper bound: 0.0544041
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550543, upper bound: 0.0536035
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552915, upper bound: 0.0546110
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0196597, 0.0224855, -0.0182549, 0.0183299, -0.0379896, 0.0407403
1: -0.0182302, 0.0495361, -0.0175492, 0.0310600, -0.0492902, 0.0670853
2: -0.0618276, 0.0383513, -0.0452677, 0.0227551, -0.0845828, 0.0836190
3: -0.0313415, 0.0676697, -0.0302430, 0.0378163, -0.0691578, 0.0979127
4: -0.0859554, 0.0461012, -0.0534865, 0.0273936, -0.1133490, 0.0995877

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552140, upper bound: 0.0544090
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552355, upper bound: 0.0544054
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0196689, 0.0225018, -0.0144043, 0.0128454, -0.0325143, 0.0369061
1: -0.0182527, 0.0495553, -0.0132008, 0.0249592, -0.0432119, 0.0627561
2: -0.0618504, 0.0383671, -0.0386996, 0.0166129, -0.0784633, 0.0770667
3: -0.0313732, 0.0676997, -0.0253240, 0.0303233, -0.0616965, 0.0930237
4: -0.0859845, 0.0461219, -0.0478803, 0.0207241, -0.1067086, 0.0940022

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551941, upper bound: 0.0544890
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552476, upper bound: 0.0545095
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0177528, 0.0200240, -0.0167507, 0.0175735, -0.0353263, 0.0367747
1: -0.0189958, 0.0363287, -0.0169593, 0.0324132, -0.0514090, 0.0532879
2: -0.0453282, 0.0254624, -0.0453734, 0.0220414, -0.0673696, 0.0708357
3: -0.0320662, 0.0460069, -0.0299786, 0.0405239, -0.0725901, 0.0759855
4: -0.0614046, 0.0284768, -0.0578682, 0.0270928, -0.0884974, 0.0863450

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556163, upper bound: 0.0550043
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0551134
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0244671, 0.0285040, -0.0174846, 0.0184832, -0.0429503, 0.0459887
1: -0.0291369, 0.0633202, -0.0177545, 0.0346891, -0.0638260, 0.0810747
2: -0.0632677, 0.0448866, -0.0469266, 0.0241803, -0.0874480, 0.0918133
3: -0.0417759, 0.0822587, -0.0308810, 0.0437489, -0.0855248, 0.1131397
4: -0.0902513, 0.0482991, -0.0604094, 0.0290925, -0.1193438, 0.1087085

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549838, upper bound: 0.0549551
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244671, 0.0285040, -0.0169136, 0.0178243, -0.0422914, 0.0454177
1: -0.0291369, 0.0633202, -0.0172633, 0.0331967, -0.0623337, 0.0805835
2: -0.0632677, 0.0448866, -0.0456765, 0.0224354, -0.0857031, 0.0905631
3: -0.0417759, 0.0822587, -0.0304100, 0.0416375, -0.0834134, 0.1126687
4: -0.0902513, 0.0482991, -0.0582723, 0.0274961, -0.1177474, 0.1065714

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0554596
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554748, upper bound: 0.0554067
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0237858, 0.0274615, -0.0163063, 0.0177085, -0.0414942, 0.0437677
1: -0.0287791, 0.0601374, -0.0184658, 0.0330927, -0.0618718, 0.0786032
2: -0.0597000, 0.0422977, -0.0422703, 0.0221831, -0.0818831, 0.0845681
3: -0.0409966, 0.0778414, -0.0314417, 0.0417711, -0.0827677, 0.1092831
4: -0.0859276, 0.0443653, -0.0568412, 0.0248886, -0.1108162, 0.1012065

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0538546, upper bound: 0.0552060
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0538281, upper bound: 0.0548437
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0193233, 0.0216105, -0.0409446, 0.0534339, -0.0727572, 0.0625551
1: -0.0221435, 0.0412708, -0.0738328, 0.1323918, -0.1545353, 0.1151036
2: -0.0484471, 0.0279795, -0.0942827, 0.0882237, -0.1366708, 0.1222622
3: -0.0359738, 0.0530184, -0.1195522, 0.1947030, -0.2306768, 0.1725707
4: -0.0654308, 0.0313087, -0.1552908, 0.1140625, -0.1794932, 0.1865995

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 38

Time for candidate selection: 5.06 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543927, upper bound: 0.0518935
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553700, upper bound: 0.0544022
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0193233, 0.0216105, -0.0388822, 0.0488454, -0.0681687, 0.0604927
1: -0.0221435, 0.0412708, -0.0675132, 0.1196551, -0.1417986, 0.1087840
2: -0.0484471, 0.0279795, -0.0903147, 0.0808700, -0.1293171, 0.1182942
3: -0.0359738, 0.0530184, -0.1089421, 0.1758024, -0.2117762, 0.1619605
4: -0.0654308, 0.0313087, -0.1444890, 0.1051642, -0.1705949, 0.1757977

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 38

Time for candidate selection: 5.07 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547440
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547734
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0356584, 0.0456048, -0.0649398, 0.0572799
1: -0.0221682, 0.0413110, -0.0588019, 0.1074852, -0.1296534, 0.1001130
2: -0.0484735, 0.0279975, -0.0863000, 0.0723548, -0.1208283, 0.1142975
3: -0.0360067, 0.0530757, -0.0943957, 0.1588094, -0.1948161, 0.1474714
4: -0.0654610, 0.0313325, -0.1380775, 0.0922416, -0.1577026, 0.1694100

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552869, upper bound: 0.0546241
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0549924
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0193233, 0.0216105, -0.0253816, 0.0294349, -0.0487582, 0.0469921
1: -0.0221435, 0.0412708, -0.0282665, 0.0643411, -0.0864847, 0.0695373
2: -0.0484471, 0.0279795, -0.0635668, 0.0452627, -0.0937098, 0.0915463
3: -0.0359738, 0.0530184, -0.0410936, 0.0820307, -0.1180045, 0.0941120
4: -0.0654308, 0.0313087, -0.0872937, 0.0487742, -0.1142050, 0.1186024

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 38

Time for candidate selection: 5.06 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546322
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546616
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0193233, 0.0216105, -0.0247393, 0.0279610, -0.0472844, 0.0463498
1: -0.0221435, 0.0412708, -0.0272721, 0.0596626, -0.0818062, 0.0685429
2: -0.0484471, 0.0279795, -0.0625226, 0.0431439, -0.0915910, 0.0905022
3: -0.0359738, 0.0530184, -0.0404192, 0.0758799, -0.1118537, 0.0934376
4: -0.0654308, 0.0313087, -0.0845790, 0.0471494, -0.1125801, 0.1158877

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38

Time for candidate selection: 4.99 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0546673
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0546967
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0191731, 0.0226172, -0.0419522, 0.0407946
1: -0.0221682, 0.0413110, -0.0188379, 0.0438878, -0.0660560, 0.0601489
2: -0.0484735, 0.0279975, -0.0516788, 0.0353989, -0.0838725, 0.0796764
3: -0.0360067, 0.0530757, -0.0302866, 0.0553400, -0.0913467, 0.0833623
4: -0.0654610, 0.0313325, -0.0726424, 0.0376871, -0.1031481, 0.1039749

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553298, upper bound: 0.0546241
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554135, upper bound: 0.0549924
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0163869, 0.0175806, -0.0397504, 0.0514172, -0.0678041, 0.0573310
1: -0.0146812, 0.0328591, -0.0689355, 0.1255305, -0.1402117, 0.1017947
2: -0.0460493, 0.0262635, -0.0941848, 0.0844139, -0.1304632, 0.1204482
3: -0.0258026, 0.0409863, -0.1116629, 0.1860772, -0.2118799, 0.1526492
4: -0.0596548, 0.0303435, -0.1541051, 0.1102471, -0.1699019, 0.1844486

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551363, upper bound: 0.0533429
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551474, upper bound: 0.0536215
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0216948, 0.0205141, -0.0413347, 0.0535006, -0.0751954, 0.0618488
1: -0.0300615, 0.0405560, -0.0737011, 0.1320493, -0.1621108, 0.1142571
2: -0.0430479, 0.0223778, -0.0967166, 0.0882339, -0.1312819, 0.1190944
3: -0.0440717, 0.0515677, -0.1197741, 0.1962066, -0.2402783, 0.1713418
4: -0.0533977, 0.0233877, -0.1596566, 0.1163448, -0.1697425, 0.1830443

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551831, upper bound: 0.0533310
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551519, upper bound: 0.0536028
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0250718, 0.0281386, -0.0414904, 0.0538612, -0.0789330, 0.0696290
1: -0.0291545, 0.0600732, -0.0740916, 0.1330158, -0.1621702, 0.1341648
2: -0.0621348, 0.0426833, -0.0972258, 0.0890399, -0.1511747, 0.1399091
3: -0.0422795, 0.0766237, -0.1205330, 0.1977145, -0.2399940, 0.1971568
4: -0.0842522, 0.0465608, -0.1608297, 0.1174413, -0.2016935, 0.2073905

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0553158
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0197673, 0.0233047, -0.0414968, 0.0538694, -0.0736368, 0.0648016
1: -0.0203831, 0.0459921, -0.0741096, 0.1330418, -0.1534249, 0.1201017
2: -0.0528844, 0.0364362, -0.0972371, 0.0890556, -0.1419400, 0.1336733
3: -0.0322222, 0.0583082, -0.1205643, 0.1977571, -0.2299793, 0.1788725
4: -0.0744631, 0.0387415, -0.1608542, 0.1174676, -0.1919307, 0.1995957

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0550682
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0553158
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0246858, 0.0310485, -0.0227199, 0.0266099, -0.0512957, 0.0537684
1: -0.0234260, 0.0675874, -0.0252418, 0.0578687, -0.0812947, 0.0928293
2: -0.0735020, 0.0570024, -0.0601313, 0.0422190, -0.1157210, 0.1171337
3: -0.0340438, 0.0885319, -0.0369541, 0.0740940, -0.1081378, 0.1254860
4: -0.1049351, 0.0622432, -0.0854829, 0.0455273, -0.1504623, 0.1477261

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552198, upper bound: 0.0542724
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551806, upper bound: 0.0546637
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0246479, 0.0285472, -0.0230395, 0.0269408, -0.0515887, 0.0515867
1: -0.0296545, 0.0637664, -0.0258892, 0.0589280, -0.0885826, 0.0896556
2: -0.0634665, 0.0448932, -0.0607148, 0.0427832, -0.1062497, 0.1056081
3: -0.0423492, 0.0829771, -0.0377389, 0.0756101, -0.1179592, 0.1207160
4: -0.0905367, 0.0483499, -0.0863544, 0.0461579, -0.1366946, 0.1347042

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546242, upper bound: 0.0550730
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555927, upper bound: 0.0555155
time: 0.39 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.06 seconds
IS_A1_A1_B1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0544190, upper bound: 0.0551188
IS_A1_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0545923, upper bound: 0.0552520
IS_A1_A1_B1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0542157, upper bound: 0.0549543
IS_A1_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0543891, upper bound: 0.0551779
IS_A1_A1_B1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0532285, upper bound: 0.0548780
IS_A1_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0544007, upper bound: 0.0552110
IS_A1_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0540543, upper bound: 0.0546370
IS_A1_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0540543, upper bound: 0.0544372
IS_A1_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0533375, upper bound: 0.0549366
IS_A1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0544968, upper bound: 0.0552202
IS_A1_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552973, upper bound: 0.0552973
IS_A1_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552973, upper bound: 0.0553991
IS_A1_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0552973
IS_A1_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0553991
IS_A1_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550043, upper bound: 0.0556163
IS_A1_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551134, upper bound: 0.0554171
IS_A1_A1_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551168, upper bound: 0.0531659
IS_A1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0556045, upper bound: 0.0550343
IS_A1_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0553164, upper bound: 0.0553232
IS_A1_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0553164, upper bound: 0.0553232
IS_A1_A1_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0545035
IS_A1_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554596, upper bound: 0.0549814
IS_A1_A1_B2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0545163, upper bound: 0.0551352
IS_A1_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552732, upper bound: 0.0553590
IS_A1_A2_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
IS_A1_A2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554519
IS_A1_A2_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
IS_A1_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555301
IS_A1_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550389, upper bound: 0.0535228
IS_A1_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552922
IS_A1_A2_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550389, upper bound: 0.0536319
IS_A1_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552922
IS_A1_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551690, upper bound: 0.0530699
IS_A1_A2_B1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551440, upper bound: 0.0532256
IS_A1_A2_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
IS_A1_A2_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
IS_A1_A2_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0534948, upper bound: 0.0550412
IS_A1_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0534948, upper bound: 0.0553158
IS_A1_A2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0518935, upper bound: 0.0543927
IS_A1_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553700
IS_A1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0547440, upper bound: 0.0555517
IS_A1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0547734, upper bound: 0.0555517
IS_A1_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0546241, upper bound: 0.0552869
IS_A1_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0554171
IS_A1_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554455
IS_A1_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554455
IS_A2_B1_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0550543, upper bound: 0.0536035
IS_A2_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552915, upper bound: 0.0546110
IS_A2_B1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552140, upper bound: 0.0544090
IS_A2_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552355, upper bound: 0.0544054
IS_A2_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551941, upper bound: 0.0544890
IS_A2_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552476, upper bound: 0.0545095
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0556163, upper bound: 0.0550043
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0551134
IS_A2_B1_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
IS_A2_B1_B1_A2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549838, upper bound: 0.0549551
IS_A2_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0554596
IS_A2_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554748, upper bound: 0.0554067
IS_A2_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0538546, upper bound: 0.0552060
IS_A2_B1_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0538281, upper bound: 0.0548437
IS_A2_B2_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0543927, upper bound: 0.0518935
IS_A2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0553700, upper bound: 0.0544022
IS_A2_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547440
IS_A2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547734
IS_A2_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552869, upper bound: 0.0546241
IS_A2_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0549924
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546322
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546616
IS_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0546673
IS_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0546967
IS_A2_B2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0553298, upper bound: 0.0546241
IS_A2_B2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554135, upper bound: 0.0549924
IS_A2_B2_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551363, upper bound: 0.0533429
IS_A2_B2_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551474, upper bound: 0.0536215
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551831, upper bound: 0.0533310
IS_A2_B2_A2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551519, upper bound: 0.0536028
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0553158
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0550682
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0553158
IS_A2_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0552198, upper bound: 0.0542724
IS_A2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0551806, upper bound: 0.0546637
IS_A2_B2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0546242, upper bound: 0.0550730
IS_A2_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 0, lower bound: -0.0555927, upper bound: 0.0555155

## BFS IS instance: IS_A1_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0253928, 0.0289675, -0.0472224, 0.0437227
1: -0.0175492, 0.0310600, -0.0249756, 0.0654672, -0.0830165, 0.0560356
2: -0.0452677, 0.0227551, -0.0737998, 0.0469031, -0.0921708, 0.0965549
3: -0.0302430, 0.0378163, -0.0401939, 0.0952320, -0.1254750, 0.0780102
4: -0.0534865, 0.0273936, -0.1105635, 0.0567574, -0.1102439, 0.1379571

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543825, upper bound: 0.0550521
time: 0.38 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543170, upper bound: 0.0552520
time: 0.40 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543170, upper bound: 0.0552520
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0167531, 0.0159926, -0.0167894, 0.0125271, -0.0292802, 0.0327820
1: -0.0151484, 0.0245952, -0.0151118, 0.0271908, -0.0423392, 0.0397071
2: -0.0405012, 0.0175003, -0.0447194, 0.0132123, -0.0537135, 0.0622196
3: -0.0271696, 0.0289076, -0.0277967, 0.0353156, -0.0624852, 0.0567043
4: -0.0452045, 0.0217376, -0.0538484, 0.0209491, -0.0661536, 0.0755860

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540700, upper bound: 0.0551254
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540700, upper bound: 0.0550713
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0182549, 0.0183299, -0.0247881, 0.0260022, -0.0442570, 0.0431181
1: -0.0175492, 0.0310600, -0.0241510, 0.0628649, -0.0804141, 0.0552109
2: -0.0452677, 0.0227551, -0.0722165, 0.0445816, -0.0898493, 0.0949716
3: -0.0302430, 0.0378163, -0.0392574, 0.0912731, -0.1215160, 0.0770736
4: -0.0534865, 0.0273936, -0.1071901, 0.0545893, -0.1080758, 0.1345837

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7

Time for candidate selection: 4.27 seconds

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541612, upper bound: 0.0551098
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0538400, upper bound: 0.0544172
time: 0.38 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542290, upper bound: 0.0548088
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.90 seconds
IS_A1_A1_B1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 7.90
Output dim: 0, lower bound: -0.0543170, upper bound: 0.0552520
IS_A1_A1_B1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 7.90
Output dim: 0, lower bound: -0.0543170, upper bound: 0.0552520
IS_A1_A1_B1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -0.0540700, upper bound: 0.0551254
IS_A1_A1_B1_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -0.0540700, upper bound: 0.0550713
IS_A1_A1_B1_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -0.0538400, upper bound: 0.0544172
IS_A1_A1_B1_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -0.0542290, upper bound: 0.0548088
IS_A1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0544968, upper bound: 0.0552202
IS_A1_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552973, upper bound: 0.0552973
IS_A1_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552973, upper bound: 0.0553991
IS_A1_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0552973
IS_A1_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0553991
IS_A1_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0550043, upper bound: 0.0556163
IS_A1_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0551134, upper bound: 0.0554171
IS_A1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0556045, upper bound: 0.0550343
IS_A1_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0553164, upper bound: 0.0553232
IS_A1_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0553164, upper bound: 0.0553232
IS_A1_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554596, upper bound: 0.0549814
IS_A1_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552732, upper bound: 0.0553590
IS_A1_A2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554519
IS_A1_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555301
IS_A1_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552922
IS_A1_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552922
IS_A1_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0551690, upper bound: 0.0530699
IS_A1_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0534948, upper bound: 0.0553158
IS_A1_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0544022, upper bound: 0.0553700
IS_A1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0547440, upper bound: 0.0555517
IS_A1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0547734, upper bound: 0.0555517
IS_A1_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0546241, upper bound: 0.0552869
IS_A1_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0554171
IS_A1_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554455
IS_A1_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554455
IS_A2_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552915, upper bound: 0.0546110
IS_A2_B1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552140, upper bound: 0.0544090
IS_A2_B1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552355, upper bound: 0.0544054
IS_A2_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0551941, upper bound: 0.0544890
IS_A2_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552476, upper bound: 0.0545095
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0556163, upper bound: 0.0550043
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0551134
IS_A2_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0554596
IS_A2_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554748, upper bound: 0.0554067
IS_A2_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0538546, upper bound: 0.0552060
IS_A2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0553700, upper bound: 0.0544022
IS_A2_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547440
IS_A2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0555517, upper bound: 0.0547734
IS_A2_B2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552869, upper bound: 0.0546241
IS_A2_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554171, upper bound: 0.0549924
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546322
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0555092, upper bound: 0.0546616
IS_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0546673
IS_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554809, upper bound: 0.0546967
IS_A2_B2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0553298, upper bound: 0.0546241
IS_A2_B2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554135, upper bound: 0.0549924
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0551831, upper bound: 0.0533310
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0553158
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0550682
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0554755, upper bound: 0.0553158
IS_A2_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0552198, upper bound: 0.0542724
IS_A2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0551806, upper bound: 0.0546637
IS_A2_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.90
Output dim: 0, lower bound: -0.0555927, upper bound: 0.0555155
Binary search (step 0): status=Status.UNKNOWN, low=0.0036636, high=0.1018318, mid=0.1018318, abs_max=0.058847926557064056
rel_dist={0: [-0.05600625688426092, 0.05600625688426092]}

## Binary search (step 1) starts
Candidate diff: 0.0527477


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0252327, 0.0271660, -0.0478260, 0.0468551
1: -0.0226827, 0.0442280, -0.0300615, 0.0599364, -0.0826191, 0.0742894
2: -0.0535189, 0.0294451, -0.0625250, 0.0373616, -0.0908805, 0.0919701
3: -0.0368305, 0.0571968, -0.0458409, 0.0809816, -0.1178121, 0.1030376
4: -0.0685483, 0.0351860, -0.0836424, 0.0439409, -0.1124892, 0.1188284

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
time: 0.33 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0252271, 0.0273805, -0.0487444, 0.0490943
1: -0.0256137, 0.0480500, -0.0305537, 0.0594033, -0.0850170, 0.0786038
2: -0.0530172, 0.0317309, -0.0617325, 0.0368690, -0.0898862, 0.0934634
3: -0.0398649, 0.0629934, -0.0464436, 0.0803663, -0.1202312, 0.1094370
4: -0.0719933, 0.0355104, -0.0826624, 0.0428012, -0.1147945, 0.1181729

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.63 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554034
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.63
Output dim: 0, lower bound: -0.0553942, upper bound: 0.0553942

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0241059, 0.0256374, -0.0430777, 0.0424940
1: -0.0179365, 0.0346897, -0.0279777, 0.0550209, -0.0729574, 0.0626674
2: -0.0468226, 0.0235523, -0.0600588, 0.0351303, -0.0819529, 0.0836111
3: -0.0312331, 0.0437208, -0.0435522, 0.0732433, -0.1044764, 0.0872731
4: -0.0597628, 0.0287614, -0.0786268, 0.0415550, -0.1013178, 0.1073882

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546562, upper bound: 0.0551484
time: 0.44 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
time: 0.36 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0242577, 0.0255441, -0.0533429, 0.0558732
1: -0.0329395, 0.0745442, -0.0281957, 0.0543739, -0.0873134, 0.1027399
2: -0.0723793, 0.0529296, -0.0596150, 0.0350196, -0.1073989, 0.1125446
3: -0.0475005, 0.0975785, -0.0435730, 0.0722541, -0.1197546, 0.1411515
4: -0.0997654, 0.0589750, -0.0770158, 0.0412395, -0.1410049, 0.1359909

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0554034
time: 0.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0206048, 0.0230341, -0.0216025, 0.0229951, -0.0435999, 0.0446365
1: -0.0241881, 0.0453783, -0.0242106, 0.0456234, -0.0698114, 0.0695889
2: -0.0513226, 0.0303424, -0.0539529, 0.0303841, -0.0817067, 0.0842953
3: -0.0383645, 0.0590674, -0.0388668, 0.0592243, -0.0975888, 0.0979342
4: -0.0694719, 0.0339743, -0.0695151, 0.0356756, -0.1051475, 0.1034894

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0207159, 0.0229785, -0.0367084, 0.0447420, -0.0654579, 0.0596868
1: -0.0242495, 0.0451704, -0.0515942, 0.1122316, -0.1364811, 0.0967646
2: -0.0509778, 0.0302205, -0.0882678, 0.0638482, -0.1148260, 0.1184883
3: -0.0382687, 0.0587770, -0.0728361, 0.1630409, -0.2013096, 0.1316131
4: -0.0686012, 0.0337570, -0.1400478, 0.0710807, -0.1396819, 0.1738049

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0552298
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0553942
time: 0.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.68 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0546562, upper bound: 0.0551484
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0553428, upper bound: 0.0554029
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0554034
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0552298
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.68
Output dim: 0, lower bound: -0.0553318, upper bound: 0.0553942

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0221450, 0.0233600, -0.0408003, 0.0405331
1: -0.0179365, 0.0346897, -0.0241089, 0.0474098, -0.0653462, 0.0587986
2: -0.0468226, 0.0235523, -0.0560589, 0.0317933, -0.0786159, 0.0796112
3: -0.0312331, 0.0437208, -0.0386023, 0.0617263, -0.0929594, 0.0823231
4: -0.0597628, 0.0287614, -0.0721812, 0.0378453, -0.0976081, 0.1009426

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553071, upper bound: 0.0549814
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552756, upper bound: 0.0553163
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0218219, 0.0229486, -0.0507474, 0.0534374
1: -0.0329395, 0.0745442, -0.0241532, 0.0465708, -0.0795103, 0.0986975
2: -0.0723793, 0.0529296, -0.0552682, 0.0311122, -0.1034916, 0.1081978
3: -0.0475005, 0.0975785, -0.0390679, 0.0604417, -0.1079422, 0.1366464
4: -0.0997654, 0.0589750, -0.0708315, 0.0371454, -0.1369109, 0.1298065

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0418720, 0.0505042, -0.0783030, 0.0734875
1: -0.0329395, 0.0745442, -0.0595902, 0.1299959, -0.1629354, 0.1341344
2: -0.0723793, 0.0529296, -0.0980435, 0.0700848, -0.1424641, 0.1509732
3: -0.0475005, 0.0975785, -0.0820598, 0.1932598, -0.2407604, 0.1796383
4: -0.0997654, 0.0589750, -0.1605624, 0.0789836, -0.1787491, 0.2195374

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553823
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553823
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0206048, 0.0230341, -0.0173510, 0.0182791, -0.0388839, 0.0403851
1: -0.0241881, 0.0453783, -0.0178209, 0.0343542, -0.0585423, 0.0631992
2: -0.0513226, 0.0303424, -0.0466004, 0.0233044, -0.0746269, 0.0769428
3: -0.0383645, 0.0590674, -0.0310903, 0.0432695, -0.0816340, 0.0901578
4: -0.0694719, 0.0339743, -0.0593948, 0.0285069, -0.0979788, 0.0933691

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551484, upper bound: 0.0546562
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0206048, 0.0230341, -0.0193350, 0.0216215, -0.0422263, 0.0423690
1: -0.0241881, 0.0453783, -0.0221682, 0.0413110, -0.0654991, 0.0675466
2: -0.0513226, 0.0303424, -0.0484735, 0.0279975, -0.0793201, 0.0788159
3: -0.0383645, 0.0590674, -0.0360067, 0.0530757, -0.0914402, 0.0950741
4: -0.0694719, 0.0339743, -0.0654610, 0.0313325, -0.1008044, 0.0994353

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0551679
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0367084, 0.0447420, -0.0640769, 0.0583298
1: -0.0221682, 0.0413110, -0.0515942, 0.1122316, -0.1343998, 0.0929053
2: -0.0484735, 0.0279975, -0.0882678, 0.0638482, -0.1123217, 0.1162654
3: -0.0360067, 0.0530757, -0.0728361, 0.1630409, -0.1990476, 0.1259118
4: -0.0654610, 0.0313325, -0.1400478, 0.0710807, -0.1365417, 0.1713804

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0367084, 0.0447420, -0.0696946, 0.0656525
1: -0.0302124, 0.0647534, -0.0515942, 0.1122316, -0.1424439, 0.1163477
2: -0.0640738, 0.0455330, -0.0882678, 0.0638482, -0.1279220, 0.1338008
3: -0.0429836, 0.0844141, -0.0728361, 0.1630409, -0.2060245, 0.1572502
4: -0.0914269, 0.0490248, -0.1400478, 0.0710807, -0.1625077, 0.1890726

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.68 seconds
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0553071, upper bound: 0.0549814
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552756, upper bound: 0.0553163
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553737
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553823
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0550899, upper bound: 0.0553823
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0551484, upper bound: 0.0546562
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552298, upper bound: 0.0553318
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0551679
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553318
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0550899
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0552189, upper bound: 0.0553391

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0172371, 0.0181486, -0.0257792, 0.0263925, -0.0436296, 0.0439279
1: -0.0174800, 0.0338637, -0.0284034, 0.0532803, -0.0707603, 0.0622671
2: -0.0463890, 0.0231817, -0.0618229, 0.0353327, -0.0817217, 0.0850047
3: -0.0305831, 0.0425467, -0.0436760, 0.0696634, -0.1002465, 0.0862228
4: -0.0591701, 0.0283288, -0.0757665, 0.0415087, -0.1006788, 0.1040953

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0189376, 0.0193207, -0.0361857, 0.0366703
1: -0.0171135, 0.0325955, -0.0193592, 0.0341959, -0.0513094, 0.0519547
2: -0.0455361, 0.0224429, -0.0479836, 0.0249779, -0.0705140, 0.0704266
3: -0.0301707, 0.0409704, -0.0325852, 0.0443274, -0.0744981, 0.0735556
4: -0.0579141, 0.0275006, -0.0599598, 0.0302096, -0.0881237, 0.0874604

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553047
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553163
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0174403, 0.0183881, -0.0461869, 0.0490558
1: -0.0329395, 0.0745442, -0.0179365, 0.0346897, -0.0676292, 0.0924807
2: -0.0723793, 0.0529296, -0.0468226, 0.0235523, -0.0959316, 0.0997523
3: -0.0475005, 0.0975785, -0.0312331, 0.0437208, -0.0912214, 0.1288116
4: -0.0997654, 0.0589750, -0.0597628, 0.0287614, -0.1285269, 0.1187378

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547730, upper bound: 0.0553161
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0552952
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0193350, 0.0216215, -0.0494203, 0.0509505
1: -0.0329395, 0.0745442, -0.0221682, 0.0413110, -0.0742506, 0.0967125
2: -0.0723793, 0.0529296, -0.0484735, 0.0279975, -0.1003769, 0.1014031
3: -0.0475005, 0.0975785, -0.0360067, 0.0530757, -0.1005763, 0.1335852
4: -0.0997654, 0.0589750, -0.0654610, 0.0313325, -0.1310980, 0.1244360

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547730, upper bound: 0.0553161
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0552991
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0399504, 0.0483450, -0.0761438, 0.0715659
1: -0.0329395, 0.0745442, -0.0570284, 0.1235986, -0.1565381, 0.1315726
2: -0.0723793, 0.0529296, -0.0939009, 0.0672521, -0.1396315, 0.1468306
3: -0.0475005, 0.0975785, -0.0790269, 0.1829200, -0.2304205, 0.1766054
4: -0.0997654, 0.0589750, -0.1531533, 0.0756163, -0.1753817, 0.2121283

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549079, upper bound: 0.0535317
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551470, upper bound: 0.0553594
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0237000, 0.0276457, -0.0554445, 0.0553156
1: -0.0329395, 0.0745442, -0.0270187, 0.0608168, -0.0937563, 0.1015629
2: -0.0723793, 0.0529296, -0.0619147, 0.0439293, -0.1163086, 0.1148443
3: -0.0475005, 0.0975785, -0.0392698, 0.0784101, -0.1259107, 0.1368483
4: -0.0997654, 0.0589750, -0.0880449, 0.0474173, -0.1471827, 0.1470199

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549079, upper bound: 0.0537096
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551470, upper bound: 0.0553594
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0189677, 0.0214014, -0.0173510, 0.0182791, -0.0372469, 0.0387524
1: -0.0208108, 0.0402559, -0.0178209, 0.0343542, -0.0551650, 0.0580768
2: -0.0480515, 0.0277686, -0.0466004, 0.0233044, -0.0713558, 0.0743690
3: -0.0341525, 0.0515311, -0.0310903, 0.0432695, -0.0774221, 0.0826214
4: -0.0653206, 0.0309637, -0.0593948, 0.0285069, -0.0938275, 0.0903585

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0553071
time: 0.31 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553163, upper bound: 0.0552756
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0193350, 0.0216215, -0.0409564, 0.0409564
1: -0.0221682, 0.0413110, -0.0221682, 0.0413110, -0.0634793, 0.0634793
2: -0.0484735, 0.0279975, -0.0484735, 0.0279975, -0.0764710, 0.0764710
3: -0.0360067, 0.0530757, -0.0360067, 0.0530757, -0.0890824, 0.0890824
4: -0.0654610, 0.0313325, -0.0654610, 0.0313325, -0.0967935, 0.0967935

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 5.09 seconds

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552029, upper bound: 0.0551061
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551263, upper bound: 0.0551030
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0193350, 0.0216215, -0.0465741, 0.0482791
1: -0.0302124, 0.0647534, -0.0221682, 0.0413110, -0.0715234, 0.0869217
2: -0.0640738, 0.0455330, -0.0484735, 0.0279975, -0.0920714, 0.0940065
3: -0.0429836, 0.0844141, -0.0360067, 0.0530757, -0.0960593, 0.1204208
4: -0.0914269, 0.0490248, -0.0654610, 0.0313325, -0.1227595, 0.1144858

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38

Time for candidate selection: 5.15 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536435, upper bound: 0.0546377
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0551043
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0414462, 0.0518560, -0.0711910, 0.0630677
1: -0.0221682, 0.0413110, -0.0627760, 0.1324598, -0.1546281, 0.1040870
2: -0.0484735, 0.0279975, -0.0970791, 0.0711206, -0.1195941, 0.1250766
3: -0.0360067, 0.0530757, -0.0889659, 0.1968575, -0.2328642, 0.1420416
4: -0.0654610, 0.0313325, -0.1602575, 0.0791221, -0.1445831, 0.1915900

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537092, upper bound: 0.0547958
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550383
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0237000, 0.0276457, -0.0469807, 0.0453215
1: -0.0221682, 0.0413110, -0.0270187, 0.0608168, -0.0829850, 0.0683298
2: -0.0484735, 0.0279975, -0.0619147, 0.0439293, -0.0924028, 0.0899122
3: -0.0360067, 0.0530757, -0.0392698, 0.0784101, -0.1144168, 0.0923455
4: -0.0654610, 0.0313325, -0.0880449, 0.0474173, -0.1128783, 0.1193774

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537092, upper bound: 0.0547958
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550383
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0414462, 0.0518560, -0.0768087, 0.0703904
1: -0.0302124, 0.0647534, -0.0627760, 0.1324598, -0.1626722, 0.1275294
2: -0.0640738, 0.0455330, -0.0970791, 0.0711206, -0.1351945, 0.1426120
3: -0.0429836, 0.0844141, -0.0889659, 0.1968575, -0.2398411, 0.1733800
4: -0.0914269, 0.0490248, -0.1602575, 0.0791221, -0.1705490, 0.2092823

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549048, upper bound: 0.0536320
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553711, upper bound: 0.0553146
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0237000, 0.0276457, -0.0525984, 0.0526442
1: -0.0302124, 0.0647534, -0.0270187, 0.0608168, -0.0910291, 0.0917722
2: -0.0640738, 0.0455330, -0.0619147, 0.0439293, -0.1080031, 0.1074476
3: -0.0429836, 0.0844141, -0.0392698, 0.0784101, -0.1213937, 0.1236839
4: -0.0914269, 0.0490248, -0.0880449, 0.0474173, -0.1388442, 0.1370697

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537094, upper bound: 0.0548394
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553711, upper bound: 0.0553146
time: 0.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.81 seconds
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553047
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553163
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0547730, upper bound: 0.0553161
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0552952
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0547730, upper bound: 0.0553161
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0552991
IS_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0549079, upper bound: 0.0535317
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0551470, upper bound: 0.0553594
IS_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0549079, upper bound: 0.0537096
IS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0551470, upper bound: 0.0553594
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0553071
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0553163, upper bound: 0.0552756
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0552029, upper bound: 0.0551061
IS_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0551263, upper bound: 0.0551030
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0536435, upper bound: 0.0546377
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0549944, upper bound: 0.0551043
IS_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0537092, upper bound: 0.0547958
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550383
IS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0537092, upper bound: 0.0547958
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550383
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0549048, upper bound: 0.0536320
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0553711, upper bound: 0.0553146
IS_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0537094, upper bound: 0.0548394
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.81
Output dim: 0, lower bound: -0.0553711, upper bound: 0.0553146

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0188422, 0.0191683, -0.0257792, 0.0263925, -0.0452347, 0.0449476
1: -0.0187277, 0.0338746, -0.0284034, 0.0532803, -0.0720079, 0.0622780
2: -0.0467068, 0.0241366, -0.0618229, 0.0353327, -0.0820395, 0.0859596
3: -0.0318076, 0.0415401, -0.0436760, 0.0696634, -0.1014710, 0.0852161
4: -0.0558105, 0.0288630, -0.0757665, 0.0415087, -0.0973192, 0.1046295

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553071, upper bound: 0.0549814
time: 0.35 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553071, upper bound: 0.0549814
time: 0.35 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0148684, 0.0150387, -0.0257792, 0.0263925, -0.0412609, 0.0408179
1: -0.0138386, 0.0261036, -0.0284034, 0.0532803, -0.0671188, 0.0545070
2: -0.0399446, 0.0177537, -0.0618229, 0.0353327, -0.0752773, 0.0795766
3: -0.0261081, 0.0320604, -0.0436760, 0.0696634, -0.0957715, 0.0757364
4: -0.0496877, 0.0220269, -0.0757665, 0.0415087, -0.0911964, 0.0977934

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0545035
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0549814
time: 0.36 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0173806, 0.0175822, -0.0344473, 0.0351133
1: -0.0171135, 0.0325955, -0.0165830, 0.0301213, -0.0472348, 0.0491785
2: -0.0455361, 0.0224429, -0.0444430, 0.0221083, -0.0676444, 0.0668859
3: -0.0301707, 0.0409704, -0.0292208, 0.0379389, -0.0681097, 0.0701912
4: -0.0579141, 0.0275006, -0.0552392, 0.0270052, -0.0849193, 0.0827398

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0552944
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553047
time: 0.35 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0357145, 0.0381162, -0.0549813, 0.0534472
1: -0.0171135, 0.0325955, -0.0479352, 0.0888816, -0.1059950, 0.0805307
2: -0.0455361, 0.0224429, -0.0860682, 0.0587934, -0.1043295, 0.1085111
3: -0.0301707, 0.0409704, -0.0673886, 0.1224924, -0.1526631, 0.1083590
4: -0.0579141, 0.0275006, -0.1161377, 0.0663504, -0.1242644, 0.1436384

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0552944
time: 0.38 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553047
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0172371, 0.0181486, -0.0459938, 0.0483348
1: -0.0329159, 0.0712517, -0.0174800, 0.0338637, -0.0667796, 0.0887317
2: -0.0701464, 0.0509891, -0.0463890, 0.0231817, -0.0933281, 0.0973781
3: -0.0480456, 0.0928499, -0.0305831, 0.0425467, -0.0905924, 0.1234330
4: -0.0942536, 0.0571052, -0.0591701, 0.0283288, -0.1225824, 0.1162753

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552259
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552952
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0168651, 0.0177327, -0.0417346, 0.0441422
1: -0.0260409, 0.0594983, -0.0171135, 0.0325955, -0.0586365, 0.0766118
2: -0.0639335, 0.0455851, -0.0455361, 0.0224429, -0.0863764, 0.0911212
3: -0.0402265, 0.0769224, -0.0301707, 0.0409704, -0.0811970, 0.1070932
4: -0.0868411, 0.0507081, -0.0579141, 0.0275006, -0.1143417, 0.1086222

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548546, upper bound: 0.0536319
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552700
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0190650, 0.0213671, -0.0492123, 0.0501627
1: -0.0329159, 0.0712517, -0.0215925, 0.0403719, -0.0732878, 0.0928442
2: -0.0701464, 0.0509891, -0.0478570, 0.0275843, -0.0977307, 0.0988461
3: -0.0480456, 0.0928499, -0.0352565, 0.0517393, -0.0997849, 0.1281064
4: -0.0942536, 0.0571052, -0.0647532, 0.0307801, -0.1250337, 0.1218584

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541497, upper bound: 0.0548856
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547730, upper bound: 0.0553161
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0187523, 0.0208727, -0.0448746, 0.0460294
1: -0.0260409, 0.0594983, -0.0209791, 0.0388048, -0.0648457, 0.0804774
2: -0.0639335, 0.0455851, -0.0470027, 0.0267317, -0.0906652, 0.0925878
3: -0.0402265, 0.0769224, -0.0347427, 0.0496625, -0.0898890, 0.1116651
4: -0.0868411, 0.0507081, -0.0631458, 0.0299530, -0.1167941, 0.1138540

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547493, upper bound: 0.0536888
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549929, upper bound: 0.0552752
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0399504, 0.0483450, -0.0756477, 0.0709827
1: -0.0318687, 0.0727345, -0.0570284, 0.1235986, -0.1554673, 0.1297629
2: -0.0712754, 0.0519350, -0.0939009, 0.0672521, -0.1385275, 0.1458359
3: -0.0462107, 0.0947695, -0.0790269, 0.1829200, -0.2291307, 0.1737964
4: -0.0982611, 0.0578686, -0.1531533, 0.0756163, -0.1738774, 0.2110219

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0553820
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552666
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0237000, 0.0276457, -0.0549484, 0.0547323
1: -0.0318687, 0.0727345, -0.0270187, 0.0608168, -0.0926855, 0.0997532
2: -0.0712754, 0.0519350, -0.0619147, 0.0439293, -0.1152047, 0.1138497
3: -0.0462107, 0.0947695, -0.0392698, 0.0784101, -0.1246209, 0.1340393
4: -0.0982611, 0.0578686, -0.0880449, 0.0474173, -0.1456784, 0.1459134

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534730, upper bound: 0.0545125
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0534730, upper bound: 0.0553594
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0233337, 0.0250260, -0.0171476, 0.0180394, -0.0413731, 0.0421737
1: -0.0255890, 0.0477269, -0.0173642, 0.0335312, -0.0591202, 0.0650910
2: -0.0562853, 0.0313383, -0.0461670, 0.0229337, -0.0792191, 0.0775053
3: -0.0402782, 0.0620095, -0.0304406, 0.0420968, -0.0823749, 0.0924502
4: -0.0714716, 0.0352104, -0.0588023, 0.0280744, -0.0995459, 0.0940127

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552185
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552756
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0157330, 0.0176015, -0.0167792, 0.0176276, -0.0333606, 0.0343808
1: -0.0154600, 0.0287473, -0.0170052, 0.0322794, -0.0477394, 0.0457525
2: -0.0401811, 0.0208104, -0.0453234, 0.0222053, -0.0623864, 0.0661338
3: -0.0275007, 0.0353552, -0.0300384, 0.0405470, -0.0680477, 0.0653936
4: -0.0527368, 0.0232645, -0.0575613, 0.0272568, -0.0799936, 0.0808258

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0551194
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0552756
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0171319, 0.0193760, -0.0193350, 0.0216215, -0.0387534, 0.0387109
1: -0.0185982, 0.0350459, -0.0221682, 0.0413110, -0.0599092, 0.0572142
2: -0.0431861, 0.0235984, -0.0484735, 0.0279975, -0.0711836, 0.0720719
3: -0.0321350, 0.0433074, -0.0360067, 0.0530757, -0.0852107, 0.0793141
4: -0.0576155, 0.0261779, -0.0654610, 0.0313325, -0.0889480, 0.0916389

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546095, upper bound: 0.0534765
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549786, upper bound: 0.0548674
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0408559, 0.0510473, -0.0703823, 0.0624774
1: -0.0221682, 0.0413110, -0.0617806, 0.1298355, -0.1520037, 0.1030916
2: -0.0484735, 0.0279975, -0.0958359, 0.0700496, -0.1185231, 0.1238335
3: -0.0360067, 0.0530757, -0.0876422, 0.1926000, -0.2286067, 0.1407180
4: -0.0654610, 0.0313325, -0.1578312, 0.0779233, -0.1433843, 0.1891637

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552929, upper bound: 0.0547418
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552752, upper bound: 0.0549929
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0230395, 0.0269408, -0.0462757, 0.0446610
1: -0.0221682, 0.0413110, -0.0258892, 0.0589280, -0.0810963, 0.0672003
2: -0.0484735, 0.0279975, -0.0607148, 0.0427832, -0.0912567, 0.0887124
3: -0.0360067, 0.0530757, -0.0377389, 0.0756101, -0.1116168, 0.0908146
4: -0.0654610, 0.0313325, -0.0863544, 0.0461579, -0.1116189, 0.1176869

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547386, upper bound: 0.0545659
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550378
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0243037, 0.0282464, -0.0414462, 0.0518560, -0.0761598, 0.0696927
1: -0.0291733, 0.0628643, -0.0627760, 0.1324598, -0.1616331, 0.1256403
2: -0.0628437, 0.0443860, -0.0970791, 0.0711206, -0.1339644, 0.1414650
3: -0.0415990, 0.0814768, -0.0889659, 0.1968575, -0.2384565, 0.1704427
4: -0.0897038, 0.0477546, -0.1602575, 0.0791221, -0.1688259, 0.2080121

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0552901
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552983, upper bound: 0.0552515
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0249527, 0.0289441, -0.0230395, 0.0269408, -0.0518934, 0.0519836
1: -0.0302124, 0.0647534, -0.0258892, 0.0589280, -0.0891404, 0.0906427
2: -0.0640738, 0.0455330, -0.0607148, 0.0427832, -0.1068570, 0.1062478
3: -0.0429836, 0.0844141, -0.0377389, 0.0756101, -0.1185937, 0.1221530
4: -0.0914269, 0.0490248, -0.0863544, 0.0461579, -0.1375849, 0.1353792

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549048, upper bound: 0.0538073
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549048, upper bound: 0.0538073
time: 0.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.64 seconds
IS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0553071, upper bound: 0.0549814
IS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0553071, upper bound: 0.0549814
IS_A1_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0550418, upper bound: 0.0545035
IS_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0549814
IS_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0552944
IS_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553047
IS_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0552944
IS_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551194, upper bound: 0.0553047
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552259
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552952
IS_A1_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548546, upper bound: 0.0536319
IS_A1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552700
IS_A1_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0541497, upper bound: 0.0548856
IS_A1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0547730, upper bound: 0.0553161
IS_A1_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0547493, upper bound: 0.0536888
IS_A1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549929, upper bound: 0.0552752
IS_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0553820
IS_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552666
IS_A1_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0534730, upper bound: 0.0545125
IS_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0534730, upper bound: 0.0553594
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552185
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552756
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0551194
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0552756
IS_A2_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0546095, upper bound: 0.0534765
IS_A2_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549786, upper bound: 0.0548674
IS_A2_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552929, upper bound: 0.0547418
IS_A2_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552752, upper bound: 0.0549929
IS_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0547386, upper bound: 0.0545659
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550378
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0552901
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552983, upper bound: 0.0552515
IS_A2_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549048, upper bound: 0.0538073
IS_A2_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549048, upper bound: 0.0538073

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0188422, 0.0191683, -0.0204502, 0.0206783, -0.0395205, 0.0396185
1: -0.0187277, 0.0338746, -0.0208089, 0.0380290, -0.0567567, 0.0546835
2: -0.0467068, 0.0241366, -0.0504493, 0.0268897, -0.0735965, 0.0745860
3: -0.0318076, 0.0415401, -0.0341723, 0.0475689, -0.0793765, 0.0757124
4: -0.0558105, 0.0288630, -0.0600661, 0.0321493, -0.0879598, 0.0889291

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553070, upper bound: 0.0551985
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0552347
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0188422, 0.0191683, -0.0223378, 0.0238199, -0.0426621, 0.0415061
1: -0.0187277, 0.0338746, -0.0229423, 0.0436428, -0.0623704, 0.0568169
2: -0.0467068, 0.0241366, -0.0541545, 0.0294658, -0.0761726, 0.0782912
3: -0.0318076, 0.0415401, -0.0365837, 0.0561969, -0.0880045, 0.0781237
4: -0.0558105, 0.0288630, -0.0682930, 0.0331831, -0.0889935, 0.0971561

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0553520
time: 0.32 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0552347
time: 0.31 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0144519, 0.0144736, -0.0257792, 0.0263925, -0.0408443, 0.0402528
1: -0.0134160, 0.0251981, -0.0284034, 0.0532803, -0.0666963, 0.0536015
2: -0.0389422, 0.0167805, -0.0618229, 0.0353327, -0.0742749, 0.0786034
3: -0.0256370, 0.0306983, -0.0436760, 0.0696634, -0.0953004, 0.0743743
4: -0.0483767, 0.0208874, -0.0757665, 0.0415087, -0.0898855, 0.0966539

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549814
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549814
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0144305, 0.0141843, -0.0310494, 0.0321632
1: -0.0171135, 0.0325955, -0.0131581, 0.0251122, -0.0422257, 0.0457536
2: -0.0455361, 0.0224429, -0.0389022, 0.0168341, -0.0623702, 0.0613451
3: -0.0301707, 0.0409704, -0.0252906, 0.0304506, -0.0606213, 0.0662610
4: -0.0579141, 0.0275006, -0.0482360, 0.0209963, -0.0789104, 0.0757366

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547467, upper bound: 0.0536153
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550816, upper bound: 0.0552701
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0142855, 0.0160951, -0.0329601, 0.0320182
1: -0.0171135, 0.0325955, -0.0128541, 0.0256413, -0.0427547, 0.0454496
2: -0.0455361, 0.0224429, -0.0365304, 0.0178321, -0.0633682, 0.0589733
3: -0.0301707, 0.0409704, -0.0242916, 0.0304201, -0.0605908, 0.0652621
4: -0.0579141, 0.0275006, -0.0479733, 0.0198085, -0.0777225, 0.0754740

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547467, upper bound: 0.0536722
time: 0.39 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550816, upper bound: 0.0552807
time: 0.39 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0338804, 0.0366893, -0.0535544, 0.0516131
1: -0.0171135, 0.0325955, -0.0452749, 0.0846471, -0.1017605, 0.0778704
2: -0.0455361, 0.0224429, -0.0824129, 0.0564430, -0.1019791, 0.1048558
3: -0.0301707, 0.0409704, -0.0641531, 0.1162573, -0.1464280, 0.1051235
4: -0.0579141, 0.0275006, -0.1115546, 0.0635662, -0.1214803, 0.1390552

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547940, upper bound: 0.0534948
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552515, upper bound: 0.0552638
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0168651, 0.0177327, -0.0193878, 0.0228643, -0.0397293, 0.0371205
1: -0.0171135, 0.0325955, -0.0188969, 0.0442249, -0.0613383, 0.0514924
2: -0.0455361, 0.0224429, -0.0520513, 0.0357740, -0.0813101, 0.0744942
3: -0.0301707, 0.0409704, -0.0305824, 0.0558650, -0.0860357, 0.0715528
4: -0.0579141, 0.0275006, -0.0731221, 0.0380710, -0.0959851, 0.1006227

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0553163
time: 0.40 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0188422, 0.0191683, -0.0470135, 0.0499399
1: -0.0329159, 0.0712517, -0.0187277, 0.0338746, -0.0667906, 0.0899794
2: -0.0701464, 0.0509891, -0.0467068, 0.0241366, -0.0942830, 0.0976960
3: -0.0480456, 0.0928499, -0.0318076, 0.0415401, -0.0895857, 0.1246575
4: -0.0942536, 0.0571052, -0.0558105, 0.0288630, -0.1231166, 0.1129157

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
time: 0.36 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0553274
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0148684, 0.0150387, -0.0428839, 0.0459661
1: -0.0329159, 0.0712517, -0.0138386, 0.0261036, -0.0590196, 0.0850903
2: -0.0701464, 0.0509891, -0.0399446, 0.0177537, -0.0879001, 0.0909337
3: -0.0480456, 0.0928499, -0.0261081, 0.0320604, -0.0801060, 0.1189580
4: -0.0942536, 0.0571052, -0.0496877, 0.0220269, -0.1162805, 0.1067929

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0553642
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0168651, 0.0177327, -0.0412201, 0.0435497
1: -0.0250003, 0.0577051, -0.0171135, 0.0325955, -0.0575959, 0.0748186
2: -0.0628079, 0.0445847, -0.0455361, 0.0224429, -0.0852508, 0.0901207
3: -0.0388779, 0.0740361, -0.0301707, 0.0409704, -0.0798483, 0.1042068
4: -0.0853493, 0.0495821, -0.0579141, 0.0275006, -0.1128500, 0.1074962

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552020
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552020
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0273508, 0.0306702, -0.0190650, 0.0213671, -0.0487179, 0.0497352
1: -0.0321755, 0.0699852, -0.0215925, 0.0403719, -0.0725474, 0.0915777
2: -0.0691589, 0.0502736, -0.0478570, 0.0275843, -0.0967432, 0.0981306
3: -0.0471808, 0.0909255, -0.0352565, 0.0517393, -0.0989201, 0.1261821
4: -0.0932161, 0.0562352, -0.0647532, 0.0307801, -0.1239962, 0.1209884

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.59 seconds

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547411, upper bound: 0.0553155
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547660, upper bound: 0.0553161
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0187523, 0.0208727, -0.0443601, 0.0454369
1: -0.0250003, 0.0577051, -0.0209791, 0.0388048, -0.0638052, 0.0786842
2: -0.0628079, 0.0445847, -0.0470027, 0.0267317, -0.0895396, 0.0915874
3: -0.0388779, 0.0740361, -0.0347427, 0.0496625, -0.0885403, 0.1087788
4: -0.0853493, 0.0495821, -0.0631458, 0.0299530, -0.1153023, 0.1127279

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545677, upper bound: 0.0547671
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0552752
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0270626, 0.0304198, -0.0398124, 0.0481918, -0.0752544, 0.0702323
1: -0.0316554, 0.0691338, -0.0567389, 0.1230701, -0.1547255, 0.1258728
2: -0.0684051, 0.0499098, -0.0936725, 0.0670874, -0.1354925, 0.1435823
3: -0.0462560, 0.0896830, -0.0786551, 0.1820644, -0.2283204, 0.1683381
4: -0.0926542, 0.0556223, -0.1526661, 0.0754223, -0.1680765, 0.2082884

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
time: 0.42 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0552666
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0390181, 0.0472205, -0.0707079, 0.0657028
1: -0.0250003, 0.0577051, -0.0552107, 0.1196632, -0.1446635, 0.1129158
2: -0.0628079, 0.0445847, -0.0922620, 0.0657265, -0.1285344, 0.1368467
3: -0.0388779, 0.0740361, -0.0767428, 0.1770525, -0.2159304, 0.1507789
4: -0.0853493, 0.0495821, -0.1497391, 0.0739088, -0.1592581, 0.1993212

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0551125
time: 0.32 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552666
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273027, 0.0310323, -0.0230395, 0.0269408, -0.0542435, 0.0540718
1: -0.0318687, 0.0727345, -0.0258892, 0.0589280, -0.0907967, 0.0986237
2: -0.0712754, 0.0519350, -0.0607148, 0.0427832, -0.1140586, 0.1126498
3: -0.0462107, 0.0947695, -0.0377389, 0.0756101, -0.1218208, 0.1325084
4: -0.0982611, 0.0578686, -0.0863544, 0.0461579, -0.1444190, 0.1442230

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527249, upper bound: 0.0553369
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0534576, upper bound: 0.0552817
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0233337, 0.0250260, -0.0187753, 0.0190549, -0.0423886, 0.0438014
1: -0.0255890, 0.0477269, -0.0186209, 0.0335240, -0.0591130, 0.0663478
2: -0.0562853, 0.0313383, -0.0465068, 0.0238822, -0.0801676, 0.0778451
3: -0.0402782, 0.0620095, -0.0316728, 0.0410670, -0.0813452, 0.0936824
4: -0.0714716, 0.0352104, -0.0554159, 0.0286110, -0.1000826, 0.0906263

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552567
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0233337, 0.0250260, -0.0148169, 0.0148870, -0.0382207, 0.0398430
1: -0.0255890, 0.0477269, -0.0137766, 0.0259240, -0.0515130, 0.0615035
2: -0.0562853, 0.0313383, -0.0397725, 0.0175647, -0.0738501, 0.0711108
3: -0.0402782, 0.0620095, -0.0260393, 0.0317957, -0.0720739, 0.0880488
4: -0.0714716, 0.0352104, -0.0493976, 0.0218287, -0.0933003, 0.0846080

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552931
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0147650, 0.0166152, -0.0167792, 0.0176276, -0.0323926, 0.0333944
1: -0.0138647, 0.0267093, -0.0170052, 0.0322794, -0.0461441, 0.0437145
2: -0.0376615, 0.0188646, -0.0453234, 0.0222053, -0.0598668, 0.0641880
3: -0.0255216, 0.0322008, -0.0300384, 0.0405470, -0.0660686, 0.0622393
4: -0.0495990, 0.0209517, -0.0575613, 0.0272568, -0.0768559, 0.0785130

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550168
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550742
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0198122, 0.0226591, -0.0167792, 0.0176276, -0.0374398, 0.0394383
1: -0.0205243, 0.0423770, -0.0170052, 0.0322794, -0.0528037, 0.0593823
2: -0.0530423, 0.0364986, -0.0453234, 0.0222053, -0.0752476, 0.0818220
3: -0.0326785, 0.0558827, -0.0300384, 0.0405470, -0.0732255, 0.0859211
4: -0.0745755, 0.0392020, -0.0575613, 0.0272568, -0.1018324, 0.0967633

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0552185
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0552185
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0190650, 0.0213671, -0.0386414, 0.0468436, -0.0659086, 0.0600085
1: -0.0215925, 0.0403719, -0.0574177, 0.1180702, -0.1396627, 0.0977896
2: -0.0478570, 0.0275843, -0.0897411, 0.0654106, -0.1132676, 0.1173254
3: -0.0352565, 0.0517393, -0.0813321, 0.1730884, -0.2083450, 0.1330714
4: -0.0647532, 0.0307801, -0.1431728, 0.0722774, -0.1370305, 0.1739530

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547824, upper bound: 0.0540679
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552929, upper bound: 0.0547418
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0187523, 0.0208727, -0.0356584, 0.0443887, -0.0631410, 0.0565311
1: -0.0209791, 0.0388048, -0.0516354, 0.1074852, -0.1284643, 0.0904402
2: -0.0470027, 0.0267317, -0.0863000, 0.0611272, -0.1081299, 0.1130316
3: -0.0347427, 0.0496625, -0.0747125, 0.1588094, -0.1935521, 0.1243750
4: -0.0631458, 0.0299530, -0.1380775, 0.0680030, -0.1311489, 0.1680305

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547671, upper bound: 0.0545677
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552752, upper bound: 0.0549924
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0193350, 0.0216215, -0.0226122, 0.0264409, -0.0457758, 0.0442337
1: -0.0221682, 0.0413110, -0.0252640, 0.0577395, -0.0799077, 0.0665750
2: -0.0484735, 0.0279975, -0.0598697, 0.0419763, -0.0904498, 0.0878673
3: -0.0360067, 0.0530757, -0.0369989, 0.0739692, -0.1099759, 0.0900747
4: -0.0654610, 0.0313325, -0.0851854, 0.0452463, -0.1107073, 0.1165180

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 38

Time for candidate selection: 4.56 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545973, upper bound: 0.0533928
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550771, upper bound: 0.0547918
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0250718, 0.0281386, -0.0412972, 0.0516869, -0.0767587, 0.0694358
1: -0.0291545, 0.0600732, -0.0624620, 0.1318737, -0.1610282, 0.1225352
2: -0.0621348, 0.0426833, -0.0968248, 0.0709320, -0.1330668, 0.1395082
3: -0.0422795, 0.0766237, -0.0885514, 0.1959007, -0.2381802, 0.1651752
4: -0.0842522, 0.0465608, -0.1597131, 0.0789044, -0.1631566, 0.2062739

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0552515
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0197673, 0.0233047, -0.0404720, 0.0506286, -0.0703960, 0.0637767
1: -0.0203831, 0.0459921, -0.0608202, 0.1282863, -0.1486694, 0.1068123
2: -0.0528844, 0.0364362, -0.0953421, 0.0694978, -0.1223822, 0.1317783
3: -0.0322222, 0.0583082, -0.0864521, 0.1905906, -0.2228128, 0.1447604
4: -0.0744631, 0.0387415, -0.1566070, 0.0773212, -0.1517843, 0.1953485

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552983, upper bound: 0.0550682
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552983, upper bound: 0.0552515
time: 0.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.24 seconds
IS_A1_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0553070, upper bound: 0.0551985
IS_A1_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0552347
IS_A1_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0553520
IS_A1_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0552347
IS_A1_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549814
IS_A1_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549814
IS_A1_A1_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547467, upper bound: 0.0536153
IS_A1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550816, upper bound: 0.0552701
IS_A1_A1_B2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547467, upper bound: 0.0536722
IS_A1_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550816, upper bound: 0.0552807
IS_A1_A1_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547940, upper bound: 0.0534948
IS_A1_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552515, upper bound: 0.0552638
IS_A1_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0553163
IS_A1_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
IS_A1_A2_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
IS_A1_A2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0553274
IS_A1_A2_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
IS_A1_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0553642
IS_A1_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552020
IS_A1_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552020
IS_A1_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547411, upper bound: 0.0553155
IS_A1_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547660, upper bound: 0.0553161
IS_A1_A2_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0545677, upper bound: 0.0547671
IS_A1_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0552752
IS_A1_A2_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
IS_A1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0552666
IS_A1_A2_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0551125
IS_A1_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552666
IS_A1_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0527249, upper bound: 0.0553369
IS_A1_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0534576, upper bound: 0.0552817
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552567
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0552931
IS_A2_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550168
IS_A2_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550742
IS_A2_B1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0552185
IS_A2_B1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0552185
IS_A2_B2_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547824, upper bound: 0.0540679
IS_A2_B2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552929, upper bound: 0.0547418
IS_A2_B2_A1_B1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0547671, upper bound: 0.0545677
IS_A2_B2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552752, upper bound: 0.0549924
IS_A2_B2_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0545973, upper bound: 0.0533928
IS_A2_B2_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0550771, upper bound: 0.0547918
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0552515
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552983, upper bound: 0.0550682
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 0, lower bound: -0.0552983, upper bound: 0.0552515

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0185753, 0.0188397, -0.0200627, 0.0202829, -0.0388582, 0.0389023
1: -0.0183657, 0.0329015, -0.0199646, 0.0369529, -0.0553187, 0.0528660
2: -0.0459855, 0.0234206, -0.0495293, 0.0265467, -0.0725322, 0.0729498
3: -0.0313385, 0.0402167, -0.0329607, 0.0462741, -0.0776126, 0.0731774
4: -0.0547361, 0.0280395, -0.0597214, 0.0314189, -0.0861551, 0.0877609

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553931, upper bound: 0.0552052
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553931, upper bound: 0.0552052
time: 0.31 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0188422, 0.0191683, -0.0198833, 0.0200708, -0.0389129, 0.0390516
1: -0.0187277, 0.0338746, -0.0200534, 0.0363166, -0.0550443, 0.0539280
2: -0.0467068, 0.0241366, -0.0491764, 0.0257585, -0.0724653, 0.0733130
3: -0.0318076, 0.0415401, -0.0332656, 0.0452280, -0.0770356, 0.0748056
4: -0.0558105, 0.0288630, -0.0584067, 0.0308806, -0.0866910, 0.0872697

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553274, upper bound: 0.0553103
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553274, upper bound: 0.0553103
time: 0.32 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0185107, 0.0187518, -0.0221287, 0.0235862, -0.0420969, 0.0408805
1: -0.0178545, 0.0326230, -0.0226675, 0.0428985, -0.0607530, 0.0552905
2: -0.0458835, 0.0238966, -0.0536185, 0.0289320, -0.0748155, 0.0775151
3: -0.0305152, 0.0399753, -0.0362537, 0.0551344, -0.0856496, 0.0762291
4: -0.0555754, 0.0282560, -0.0674973, 0.0325831, -0.0881585, 0.0957533

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0551986
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0552347
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0183495, 0.0186593, -0.0223378, 0.0238199, -0.0421694, 0.0409971
1: -0.0180988, 0.0325174, -0.0229423, 0.0436428, -0.0617415, 0.0554597
2: -0.0456146, 0.0231816, -0.0541545, 0.0294658, -0.0750804, 0.0773362
3: -0.0310402, 0.0397046, -0.0365837, 0.0561969, -0.0872371, 0.0762883
4: -0.0545507, 0.0277470, -0.0682930, 0.0331831, -0.0877338, 0.0960400

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0551986
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552567, upper bound: 0.0552347
time: 0.33 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0144519, 0.0144736, -0.0204502, 0.0206783, -0.0351302, 0.0349237
1: -0.0134160, 0.0251981, -0.0208089, 0.0380290, -0.0514451, 0.0460070
2: -0.0389422, 0.0167805, -0.0504493, 0.0268897, -0.0658319, 0.0672298
3: -0.0256370, 0.0306983, -0.0341723, 0.0475689, -0.0732058, 0.0648706
4: -0.0483767, 0.0208874, -0.0600661, 0.0321493, -0.0805261, 0.0809535

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549463
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549814
time: 0.40 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0144519, 0.0144736, -0.0223378, 0.0238199, -0.0382718, 0.0368113
1: -0.0134160, 0.0251981, -0.0229423, 0.0436428, -0.0570588, 0.0481404
2: -0.0389422, 0.0167805, -0.0541545, 0.0294658, -0.0684080, 0.0709350
3: -0.0256370, 0.0306983, -0.0365837, 0.0561969, -0.0818339, 0.0672819
4: -0.0483767, 0.0208874, -0.0682930, 0.0331831, -0.0815598, 0.0891804

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549465
time: 0.40 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552931, upper bound: 0.0549814
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0161526, 0.0169241, -0.0144305, 0.0141843, -0.0303370, 0.0313546
1: -0.0161411, 0.0303623, -0.0131581, 0.0251122, -0.0412533, 0.0435204
2: -0.0440827, 0.0209353, -0.0389022, 0.0168341, -0.0609168, 0.0598374
3: -0.0289245, 0.0379305, -0.0252906, 0.0304506, -0.0593751, 0.0632211
4: -0.0560189, 0.0258288, -0.0482360, 0.0209963, -0.0770153, 0.0740648

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552017, upper bound: 0.0552701
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552017, upper bound: 0.0552017
time: 0.40 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0161526, 0.0169241, -0.0142855, 0.0160951, -0.0322477, 0.0312096
1: -0.0161411, 0.0303623, -0.0128541, 0.0256413, -0.0417823, 0.0432164
2: -0.0440827, 0.0209353, -0.0365304, 0.0178321, -0.0619148, 0.0574657
3: -0.0289245, 0.0379305, -0.0242916, 0.0304201, -0.0593446, 0.0622221
4: -0.0560189, 0.0258288, -0.0479733, 0.0198085, -0.0758274, 0.0738021

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545677, upper bound: 0.0546909
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550816, upper bound: 0.0552807
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0161526, 0.0169241, -0.0338804, 0.0366893, -0.0528420, 0.0508045
1: -0.0161411, 0.0303623, -0.0452749, 0.0846471, -0.1007881, 0.0756372
2: -0.0440827, 0.0209353, -0.0824129, 0.0564430, -0.1005258, 0.1033481
3: -0.0289245, 0.0379305, -0.0641531, 0.1162573, -0.1451817, 0.1020836
4: -0.0560189, 0.0258288, -0.1115546, 0.0635662, -0.1195851, 0.1373834

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552020, upper bound: 0.0552638
time: 0.39 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552020, upper bound: 0.0550343
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0188422, 0.0191683, -0.0193878, 0.0228643, -0.0417065, 0.0385561
1: -0.0187277, 0.0338746, -0.0188969, 0.0442249, -0.0629525, 0.0527715
2: -0.0467068, 0.0241366, -0.0520513, 0.0357740, -0.0824808, 0.0761879
3: -0.0318076, 0.0415401, -0.0305824, 0.0558650, -0.0876726, 0.0721225
4: -0.0558105, 0.0288630, -0.0731221, 0.0380710, -0.0938815, 0.1019851

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0548130
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0553163
time: 0.41 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0148684, 0.0150387, -0.0193878, 0.0228643, -0.0377327, 0.0344265
1: -0.0138386, 0.0261036, -0.0188969, 0.0442249, -0.0580634, 0.0450005
2: -0.0399446, 0.0177537, -0.0520513, 0.0357740, -0.0757186, 0.0698050
3: -0.0261081, 0.0320604, -0.0305824, 0.0558650, -0.0819731, 0.0626428
4: -0.0496877, 0.0220269, -0.0731221, 0.0380710, -0.0877586, 0.0951490

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0545035
time: 0.38 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552185, upper bound: 0.0549814
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0183495, 0.0186593, -0.0465045, 0.0494472
1: -0.0329159, 0.0712517, -0.0180988, 0.0325174, -0.0654334, 0.0893505
2: -0.0701464, 0.0509891, -0.0456146, 0.0231816, -0.0933280, 0.0966038
3: -0.0480456, 0.0928499, -0.0310402, 0.0397046, -0.0877502, 0.1238901
4: -0.0942536, 0.0571052, -0.0545507, 0.0277470, -0.1220006, 0.1116559

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552052, upper bound: 0.0553274
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552052, upper bound: 0.0553274
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0144519, 0.0144736, -0.0423187, 0.0455496
1: -0.0329159, 0.0712517, -0.0134160, 0.0251981, -0.0581141, 0.0846677
2: -0.0701464, 0.0509891, -0.0389422, 0.0167805, -0.0869269, 0.0899313
3: -0.0480456, 0.0928499, -0.0256370, 0.0306983, -0.0787439, 0.1184869
4: -0.0942536, 0.0571052, -0.0483767, 0.0208874, -0.1151410, 0.1054820

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529221, upper bound: 0.0542551
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0553429
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0188422, 0.0191683, -0.0426557, 0.0455268
1: -0.0250003, 0.0577051, -0.0187277, 0.0338746, -0.0588749, 0.0764328
2: -0.0628079, 0.0445847, -0.0467068, 0.0241366, -0.0869445, 0.0912915
3: -0.0388779, 0.0740361, -0.0318076, 0.0415401, -0.0804179, 0.1058437
4: -0.0853493, 0.0495821, -0.0558105, 0.0288630, -0.1142123, 0.1053926

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546909, upper bound: 0.0549097
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552020
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0148684, 0.0150387, -0.0385260, 0.0415530
1: -0.0250003, 0.0577051, -0.0138386, 0.0261036, -0.0511040, 0.0715437
2: -0.0628079, 0.0445847, -0.0399446, 0.0177537, -0.0805616, 0.0845293
3: -0.0388779, 0.0740361, -0.0261081, 0.0320604, -0.0709383, 0.1001442
4: -0.0853493, 0.0495821, -0.0496877, 0.0220269, -0.1073762, 0.0992697

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546909, upper bound: 0.0549097
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0552020
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0273508, 0.0306702, -0.0169507, 0.0192059, -0.0465567, 0.0476209
1: -0.0321755, 0.0699852, -0.0182366, 0.0344253, -0.0666008, 0.0882218
2: -0.0691589, 0.0502736, -0.0427881, 0.0233385, -0.0924974, 0.0930617
3: -0.0471808, 0.0909255, -0.0316864, 0.0424571, -0.0896379, 0.1226119
4: -0.0932161, 0.0562352, -0.0571530, 0.0258345, -0.1190506, 0.1133883

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539429, upper bound: 0.0549836
time: 0.34 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546313, upper bound: 0.0552269
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0273508, 0.0306702, -0.0184639, 0.0207901, -0.0481410, 0.0491340
1: -0.0321755, 0.0699852, -0.0208705, 0.0383938, -0.0705693, 0.0908557
2: -0.0691589, 0.0502736, -0.0458396, 0.0263048, -0.0954637, 0.0961132
3: -0.0471808, 0.0909255, -0.0343252, 0.0489583, -0.0961391, 0.1252507
4: -0.0932161, 0.0562352, -0.0622887, 0.0288511, -0.1220672, 0.1185239

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541282, upper bound: 0.0549846
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546608, upper bound: 0.0552269
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0230536, 0.0262401, -0.0187523, 0.0208727, -0.0439263, 0.0449924
1: -0.0242822, 0.0563962, -0.0209791, 0.0388048, -0.0630871, 0.0773753
2: -0.0619228, 0.0438159, -0.0470027, 0.0267317, -0.0886545, 0.0908186
3: -0.0380059, 0.0721582, -0.0347427, 0.0496625, -0.0876684, 0.1069009
4: -0.0842291, 0.0486820, -0.0631458, 0.0299530, -0.1141821, 0.1118278

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 10

Time for candidate selection: 4.40 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549827, upper bound: 0.0550259
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_A2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549827, upper bound: 0.0552752
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0270626, 0.0304198, -0.0349757, 0.0422462, -0.0693088, 0.0653955
1: -0.0316554, 0.0691338, -0.0480072, 0.1026535, -0.1343089, 0.1171411
2: -0.0684051, 0.0499098, -0.0848728, 0.0588492, -0.1272543, 0.1347826
3: -0.0462560, 0.0896830, -0.0678582, 0.1512295, -0.1974856, 0.1575412
4: -0.0926542, 0.0556223, -0.1343374, 0.0662057, -0.1588599, 0.1899597

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0553793
time: 0.37 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0553429
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0230536, 0.0262401, -0.0390181, 0.0472205, -0.0702741, 0.0652583
1: -0.0242822, 0.0563962, -0.0552107, 0.1196632, -0.1439454, 0.1116069
2: -0.0619228, 0.0438159, -0.0922620, 0.0657265, -0.1276494, 0.1360778
3: -0.0380059, 0.0721582, -0.0767428, 0.1770525, -0.2150584, 0.1489010
4: -0.0842291, 0.0486820, -0.1497391, 0.0739088, -0.1581379, 0.1984210

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0550682
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552638, upper bound: 0.0550682
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0270626, 0.0304198, -0.0229275, 0.0268549, -0.0539175, 0.0533473
1: -0.0316554, 0.0691338, -0.0256542, 0.0586205, -0.0902760, 0.0947880
2: -0.0684051, 0.0499098, -0.0605550, 0.0426615, -0.1110666, 0.1104648
3: -0.0462560, 0.0896830, -0.0374802, 0.0751542, -0.1214103, 0.1271632
4: -0.0926542, 0.0556223, -0.0861181, 0.0460232, -0.1386773, 0.1417404

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0549385
time: 0.40 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0552817
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0222145, 0.0261334, -0.0496208, 0.0488992
1: -0.0250003, 0.0577051, -0.0245260, 0.0560932, -0.0810935, 0.0822311
2: -0.0628079, 0.0445847, -0.0590930, 0.0414559, -0.1042638, 0.1036776
3: -0.0388779, 0.0740361, -0.0363050, 0.0717416, -0.1106195, 0.1103411
4: -0.0853493, 0.0495821, -0.0839071, 0.0446402, -0.1299895, 0.1334891

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0551444
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550057, upper bound: 0.0552817
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0233337, 0.0250260, -0.0183251, 0.0186145, -0.0419482, 0.0433511
1: -0.0255890, 0.0477269, -0.0180574, 0.0323794, -0.0579684, 0.0657843
2: -0.0562853, 0.0313383, -0.0455349, 0.0230801, -0.0793654, 0.0768733
3: -0.0402782, 0.0620095, -0.0309879, 0.0395181, -0.0797963, 0.0929975
4: -0.0714716, 0.0352104, -0.0543937, 0.0276460, -0.0991176, 0.0896042

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551986, upper bound: 0.0552567
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551986, upper bound: 0.0552567
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0233337, 0.0250260, -0.0144423, 0.0144348, -0.0377685, 0.0394683
1: -0.0255890, 0.0477269, -0.0134045, 0.0251642, -0.0507533, 0.0611313
2: -0.0562853, 0.0313383, -0.0389097, 0.0167449, -0.0730303, 0.0702481
3: -0.0402782, 0.0620095, -0.0256239, 0.0306482, -0.0709264, 0.0876334
4: -0.0714716, 0.0352104, -0.0483217, 0.0208501, -0.0923217, 0.0835322

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549465, upper bound: 0.0552931
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549465, upper bound: 0.0552931
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0147650, 0.0166152, -0.0187753, 0.0190549, -0.0338199, 0.0353905
1: -0.0138647, 0.0267093, -0.0186209, 0.0335240, -0.0473887, 0.0453302
2: -0.0376615, 0.0188646, -0.0465068, 0.0238822, -0.0615438, 0.0653714
3: -0.0255216, 0.0322008, -0.0316728, 0.0410670, -0.0665887, 0.0638737
4: -0.0495990, 0.0209517, -0.0554159, 0.0286110, -0.0782100, 0.0763675

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0546167
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550168
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0147650, 0.0166152, -0.0148169, 0.0148870, -0.0296520, 0.0314321
1: -0.0138647, 0.0267093, -0.0137766, 0.0259240, -0.0397887, 0.0404859
2: -0.0376615, 0.0188646, -0.0397725, 0.0175647, -0.0552263, 0.0586370
3: -0.0255216, 0.0322008, -0.0260393, 0.0317957, -0.0573173, 0.0582401
4: -0.0495990, 0.0209517, -0.0493976, 0.0218287, -0.0714277, 0.0703493

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0546213
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553047, upper bound: 0.0550742
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0198122, 0.0226591, -0.0187753, 0.0190549, -0.0388671, 0.0414344
1: -0.0205243, 0.0423770, -0.0186209, 0.0335240, -0.0540483, 0.0609980
2: -0.0530423, 0.0364986, -0.0465068, 0.0238822, -0.0769245, 0.0830054
3: -0.0326785, 0.0558827, -0.0316728, 0.0410670, -0.0737455, 0.0875555
4: -0.0745755, 0.0392020, -0.0554159, 0.0286110, -0.1031865, 0.0946178

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548663, upper bound: 0.0543021
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551757, upper bound: 0.0550651
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0198122, 0.0226591, -0.0148169, 0.0148870, -0.0346992, 0.0374760
1: -0.0205243, 0.0423770, -0.0137766, 0.0259240, -0.0464483, 0.0561536
2: -0.0530423, 0.0364986, -0.0397725, 0.0175647, -0.0706070, 0.0762711
3: -0.0326785, 0.0558827, -0.0260393, 0.0317957, -0.0644742, 0.0819220
4: -0.0745755, 0.0392020, -0.0493976, 0.0218287, -0.0964042, 0.0885996

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548663, upper bound: 0.0543021
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551757, upper bound: 0.0550651
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0190650, 0.0213671, -0.0382109, 0.0462629, -0.0653279, 0.0595780
1: -0.0215925, 0.0403719, -0.0566928, 0.1164287, -0.1380212, 0.0970647
2: -0.0478570, 0.0275843, -0.0888770, 0.0646978, -0.1125548, 0.1164613
3: -0.0352565, 0.0517393, -0.0804257, 0.1704520, -0.2057085, 0.1321650
4: -0.0647532, 0.0307801, -0.1415852, 0.0714636, -0.1362167, 0.1723653

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 38

Time for candidate selection: 4.38 seconds

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552922, upper bound: 0.0547078
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552929, upper bound: 0.0547263
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0187523, 0.0208727, -0.0351976, 0.0437966, -0.0625489, 0.0560704
1: -0.0209791, 0.0388048, -0.0508207, 0.1057632, -0.1267423, 0.0896256
2: -0.0470027, 0.0267317, -0.0853627, 0.0603468, -0.1073495, 0.1120944
3: -0.0347427, 0.0496625, -0.0736660, 0.1560170, -0.1907597, 0.1233285
4: -0.0631458, 0.0299530, -0.1363904, 0.0670995, -0.1302453, 0.1663434

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 7

Time for candidate selection: 4.33 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550259, upper bound: 0.0549827
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552752, upper bound: 0.0549827
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0250718, 0.0281386, -0.0362481, 0.0451842, -0.0702560, 0.0643866
1: -0.0291545, 0.0600732, -0.0526532, 0.1100473, -0.1392018, 0.1127263
2: -0.0621348, 0.0426833, -0.0875491, 0.0621852, -0.1243200, 0.1302324
3: -0.0422795, 0.0766237, -0.0760485, 0.1631809, -0.2054604, 0.1526722
4: -0.0842522, 0.0465608, -0.1404332, 0.0692042, -0.1534564, 0.1869940

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548701, upper bound: 0.0550328
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0552714
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0197673, 0.0233047, -0.0393196, 0.0478285, -0.0675958, 0.0626244
1: -0.0203831, 0.0459921, -0.0586278, 0.1212834, -0.1416666, 0.1046199
2: -0.0528844, 0.0364362, -0.0911857, 0.0665731, -0.1194575, 0.1276219
3: -0.0322222, 0.0583082, -0.0829447, 0.1783789, -0.2106011, 0.1412530
4: -0.0744631, 0.0387415, -0.1460499, 0.0737249, -0.1481880, 0.1847914

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548784, upper bound: 0.0543360
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551463, upper bound: 0.0549177
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0197673, 0.0233047, -0.0362481, 0.0451842, -0.0649516, 0.0595528
1: -0.0203831, 0.0459921, -0.0526532, 0.1100473, -0.1304304, 0.0986453
2: -0.0528844, 0.0364362, -0.0875491, 0.0621852, -0.1150697, 0.1239853
3: -0.0322222, 0.0583082, -0.0760485, 0.1631809, -0.1954031, 0.1343567
4: -0.0744631, 0.0387415, -0.1404332, 0.0692042, -0.1436673, 0.1791747

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0036636, high=0.0527477, mid=0.0527477, abs_max=0.058847926557064056
rel_dist={0: [-0.05567830804756887, 0.055678308047568875]}

## Binary search (step 2) starts
Candidate diff: 0.0282057


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552439
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0552439, upper bound: 0.0552439

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0206600, 0.0216224, -0.0232866, 0.0246099, -0.0452699, 0.0449090
1: -0.0226827, 0.0442280, -0.0266501, 0.0523549, -0.0750376, 0.0708781
2: -0.0535189, 0.0294451, -0.0586758, 0.0341689, -0.0876878, 0.0881208
3: -0.0368305, 0.0571968, -0.0416921, 0.0688619, -0.1056924, 0.0988889
4: -0.0685483, 0.0351860, -0.0759283, 0.0403609, -0.1089091, 0.1111142

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
time: 0.28 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0213640, 0.0238672, -0.0234919, 0.0252934, -0.0466574, 0.0473591
1: -0.0256137, 0.0480500, -0.0277824, 0.0525645, -0.0781783, 0.0758324
2: -0.0530172, 0.0317309, -0.0576905, 0.0337116, -0.0867288, 0.0894214
3: -0.0398649, 0.0629934, -0.0431061, 0.0695643, -0.1094292, 0.1060995
4: -0.0719933, 0.0355104, -0.0755393, 0.0389305, -0.1109238, 0.1110497

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0552439
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.44 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552401
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0552156, upper bound: 0.0552486
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0552439

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0174403, 0.0183881, -0.0212590, 0.0224313, -0.0398715, 0.0396471
1: -0.0179365, 0.0346897, -0.0233526, 0.0456255, -0.0635620, 0.0580423
2: -0.0468226, 0.0235523, -0.0542744, 0.0302711, -0.0770938, 0.0778267
3: -0.0312331, 0.0437208, -0.0379559, 0.0590037, -0.0902368, 0.0816767
4: -0.0597628, 0.0287614, -0.0696705, 0.0361769, -0.0959397, 0.0984320

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551909, upper bound: 0.0549814
time: 0.28 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551749, upper bound: 0.0551961
time: 0.30 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0277988, 0.0316155, -0.0218001, 0.0226362, -0.0504350, 0.0534156
1: -0.0329395, 0.0745442, -0.0239870, 0.0457203, -0.0786599, 0.0985312
2: -0.0723793, 0.0529296, -0.0541115, 0.0307259, -0.1031052, 0.1070412
3: -0.0475005, 0.0975785, -0.0382393, 0.0591776, -0.1066782, 0.1358179
4: -0.0997654, 0.0589750, -0.0683273, 0.0363176, -0.1360830, 0.1273023

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
time: 0.31 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551749, upper bound: 0.0551962
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0201634, 0.0225399, -0.0200192, 0.0215588, -0.0417222, 0.0425591
1: -0.0234442, 0.0439457, -0.0220961, 0.0411768, -0.0646210, 0.0660418
2: -0.0503386, 0.0295319, -0.0501842, 0.0277946, -0.0781332, 0.0797161
3: -0.0375014, 0.0569067, -0.0360950, 0.0528612, -0.0903626, 0.0930016
4: -0.0680718, 0.0330660, -0.0649615, 0.0323495, -0.1004212, 0.0980275

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544436, upper bound: 0.0546103
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0204120, 0.0225449, -0.0313514, 0.0346807, -0.0550927, 0.0538964
1: -0.0236487, 0.0436984, -0.0411882, 0.0823566, -0.1060053, 0.0848867
2: -0.0500013, 0.0294407, -0.0775521, 0.0542020, -0.1042033, 0.1069927
3: -0.0375547, 0.0566226, -0.0578171, 0.1109901, -0.1485448, 0.1144397
4: -0.0667728, 0.0328598, -0.1076526, 0.0603157, -0.1270885, 0.1405123

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.32 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551938, upper bound: 0.0551938
time: 0.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.72 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0551909, upper bound: 0.0549814
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0551749, upper bound: 0.0551961
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0551749, upper bound: 0.0551962
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0544436, upper bound: 0.0546103
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.0551938, upper bound: 0.0551938

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0171354, 0.0180253, -0.0239053, 0.0243909, -0.0415263, 0.0419306
1: -0.0172313, 0.0334390, -0.0259228, 0.0482863, -0.0655176, 0.0593618
2: -0.0461896, 0.0229987, -0.0575209, 0.0323608, -0.0785504, 0.0805196
3: -0.0302263, 0.0419284, -0.0406964, 0.0620064, -0.0922327, 0.0826249
4: -0.0589011, 0.0281245, -0.0697371, 0.0381188, -0.0970199, 0.0978616

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0549814
time: 0.32 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0549814
time: 0.30 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0162888, 0.0169795, -0.0180836, 0.0184720, -0.0347607, 0.0350631
1: -0.0162230, 0.0302493, -0.0186409, 0.0327796, -0.0490027, 0.0488902
2: -0.0440542, 0.0211721, -0.0463854, 0.0236174, -0.0676716, 0.0675575
3: -0.0290382, 0.0379892, -0.0320357, 0.0421831, -0.0712213, 0.0700249
4: -0.0557671, 0.0260544, -0.0578919, 0.0287058, -0.0844729, 0.0839464

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551900
time: 0.32 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551961
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0212617, 0.0219966, -0.0498417, 0.0523594
1: -0.0329159, 0.0712517, -0.0228141, 0.0434672, -0.0763831, 0.0940658
2: -0.0701464, 0.0509891, -0.0529004, 0.0297213, -0.0998677, 0.1038896
3: -0.0480456, 0.0928499, -0.0366397, 0.0558769, -0.1039225, 0.1294896
4: -0.0942536, 0.0571052, -0.0666887, 0.0351450, -0.1293986, 0.1237940

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
time: 0.31 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
time: 0.28 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0203617, 0.0209416, -0.0449435, 0.0476389
1: -0.0260409, 0.0594983, -0.0219557, 0.0400879, -0.0661288, 0.0814540
2: -0.0639335, 0.0455851, -0.0506761, 0.0278381, -0.0917716, 0.0962612
3: -0.0402265, 0.0769224, -0.0356923, 0.0515754, -0.0918019, 0.1126148
4: -0.0868411, 0.0507081, -0.0632672, 0.0330553, -0.1198964, 0.1139754

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551798
time: 0.30 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551860
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0185517, 0.0209276, -0.0198868, 0.0214108, -0.0399624, 0.0408144
1: -0.0201721, 0.0389028, -0.0218420, 0.0407220, -0.0608942, 0.0607448
2: -0.0471220, 0.0269722, -0.0499137, 0.0275575, -0.0746795, 0.0768859
3: -0.0333916, 0.0495964, -0.0357575, 0.0522313, -0.0856229, 0.0853539
4: -0.0639836, 0.0301121, -0.0646016, 0.0320915, -0.0960751, 0.0947137

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0198782, 0.0220198, -0.0324696, 0.0342210, -0.0540992, 0.0544894
1: -0.0225352, 0.0418041, -0.0413091, 0.0795288, -0.1020640, 0.0831133
2: -0.0487399, 0.0285888, -0.0774679, 0.0514434, -0.1001834, 0.1060567
3: -0.0360975, 0.0538204, -0.0577321, 0.1064819, -0.1425794, 0.1115526
4: -0.0652745, 0.0317328, -0.1015937, 0.0578554, -0.1231299, 0.1333265

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.32 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0190364, 0.0208087, -0.0272329, 0.0298646, -0.0489009, 0.0480416
1: -0.0209704, 0.0378331, -0.0333589, 0.0658248, -0.0867952, 0.0711920
2: -0.0467835, 0.0264615, -0.0680115, 0.0461544, -0.0929379, 0.0944730
3: -0.0346809, 0.0484497, -0.0491278, 0.0880529, -0.1227338, 0.0975775
4: -0.0614633, 0.0296071, -0.0925316, 0.0512802, -0.1127435, 0.1221387

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544111, upper bound: 0.0546565
time: 0.33 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550315, upper bound: 0.0550315
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.71 seconds
IS_A1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0549814
IS_A1_A1_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0551480, upper bound: 0.0549814
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551900
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551961
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551798
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551860
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0551508, upper bound: 0.0551956
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
IS_A2_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0544111, upper bound: 0.0546565
IS_A2_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -0.0550315, upper bound: 0.0550315

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0162888, 0.0169795, -0.0169558, 0.0172645, -0.0335533, 0.0339353
1: -0.0162230, 0.0302493, -0.0167262, 0.0297195, -0.0459425, 0.0469756
2: -0.0440542, 0.0211721, -0.0439008, 0.0215698, -0.0656240, 0.0650730
3: -0.0290382, 0.0379892, -0.0297460, 0.0374564, -0.0664946, 0.0677352
4: -0.0557671, 0.0260544, -0.0547298, 0.0263752, -0.0821423, 0.0807842

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551900
time: 0.32 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551900
time: 0.32 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0162888, 0.0169795, -0.0350453, 0.0362561, -0.0525448, 0.0520248
1: -0.0162230, 0.0302493, -0.0464443, 0.0840745, -0.1002976, 0.0766936
2: -0.0440542, 0.0211721, -0.0838097, 0.0569665, -0.1010207, 0.1049818
3: -0.0290382, 0.0379892, -0.0660995, 0.1159469, -0.1449850, 0.1040887
4: -0.0557671, 0.0260544, -0.1109359, 0.0650300, -0.1207971, 0.1369903

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546628, upper bound: 0.0536641
time: 0.32 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551723
time: 0.32 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0188964, 0.0193233, -0.0471685, 0.0499941
1: -0.0329159, 0.0712517, -0.0194188, 0.0365750, -0.0694909, 0.0906705
2: -0.0701464, 0.0509891, -0.0484481, 0.0256138, -0.0957602, 0.0994372
3: -0.0480456, 0.0928499, -0.0321149, 0.0464251, -0.0944707, 0.1249648
4: -0.0942536, 0.0571052, -0.0605656, 0.0306382, -0.1248918, 0.1176708

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549968, upper bound: 0.0551374
time: 0.30 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552237
time: 0.31 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0197716, 0.0219174, -0.0497626, 0.0508694
1: -0.0329159, 0.0712517, -0.0223491, 0.0414975, -0.0744135, 0.0936008
2: -0.0701464, 0.0509891, -0.0484897, 0.0284146, -0.0985610, 0.0994788
3: -0.0480456, 0.0928499, -0.0358685, 0.0533735, -0.1014191, 0.1287184
4: -0.0942536, 0.0571052, -0.0650338, 0.0315085, -0.1257621, 0.1221390

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
time: 0.30 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
time: 0.30 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0185063, 0.0192717, -0.0432736, 0.0457834
1: -0.0260409, 0.0594983, -0.0192202, 0.0357651, -0.0618061, 0.0787185
2: -0.0639335, 0.0455851, -0.0481066, 0.0250167, -0.0889502, 0.0936917
3: -0.0402265, 0.0769224, -0.0328924, 0.0456334, -0.0858600, 0.1098149
4: -0.0868411, 0.0507081, -0.0607494, 0.0303771, -0.1172182, 0.1114576

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551798
time: 0.30 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551798
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0384816, 0.0410192, -0.0650211, 0.0657587
1: -0.0260409, 0.0594983, -0.0531615, 0.0996819, -0.1257229, 0.1126598
2: -0.0639335, 0.0455851, -0.0913700, 0.0636415, -0.1275750, 0.1369551
3: -0.0402265, 0.0769224, -0.0736492, 0.1377085, -0.1779350, 0.1505717
4: -0.0868411, 0.0507081, -0.1246410, 0.0717990, -0.1586401, 0.1753491

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551860
time: 0.32 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551860
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0185517, 0.0209276, -0.0173009, 0.0182213, -0.0367730, 0.0382285
1: -0.0201721, 0.0389028, -0.0177054, 0.0341504, -0.0543225, 0.0566082
2: -0.0471220, 0.0269722, -0.0465042, 0.0232183, -0.0703403, 0.0734764
3: -0.0333916, 0.0495964, -0.0309278, 0.0429934, -0.0763850, 0.0805242
4: -0.0639836, 0.0301121, -0.0592682, 0.0284133, -0.0923969, 0.0893802

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551098
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0185517, 0.0209276, -0.0191889, 0.0214792, -0.0400308, 0.0401165
1: -0.0201721, 0.0389028, -0.0218908, 0.0408736, -0.0610457, 0.0607936
2: -0.0471220, 0.0269722, -0.0481801, 0.0277699, -0.0748920, 0.0751523
3: -0.0333916, 0.0495964, -0.0356535, 0.0524579, -0.0858495, 0.0852499
4: -0.0639836, 0.0301121, -0.0651084, 0.0310635, -0.0950471, 0.0952205

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551098
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0198782, 0.0220198, -0.0380102, 0.0402471, -0.0601253, 0.0600299
1: -0.0225352, 0.0418041, -0.0539456, 0.0992521, -0.1217873, 0.0957497
2: -0.0487399, 0.0285888, -0.0882765, 0.0612784, -0.1100184, 0.1168652
3: -0.0360975, 0.0538204, -0.0748796, 0.1375122, -0.1736097, 0.1287001
4: -0.0652745, 0.0317328, -0.1215643, 0.0684904, -0.1337649, 0.1532971

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.35 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0198782, 0.0220198, -0.0253017, 0.0285668, -0.0484450, 0.0473215
1: -0.0225352, 0.0418041, -0.0282283, 0.0612988, -0.0838340, 0.0700324
2: -0.0487399, 0.0285888, -0.0637410, 0.0433148, -0.0920547, 0.0923297
3: -0.0360975, 0.0538204, -0.0414863, 0.0783977, -0.1144952, 0.0953068
4: -0.0652745, 0.0317328, -0.0863181, 0.0472957, -0.1125703, 0.1180509

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.34 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.74 seconds
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551900
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550687, upper bound: 0.0551900
IS_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0546628, upper bound: 0.0536641
IS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551723
IS_A1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0549968, upper bound: 0.0551374
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552237
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552298
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551798
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551798
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551860
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0550370, upper bound: 0.0551860
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551098
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551098
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0551387, upper bound: 0.0551956
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157
IS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.74
Output dim: 0, lower bound: -0.0552148, upper bound: 0.0550157

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0162888, 0.0169795, -0.0148684, 0.0150387, -0.0313274, 0.0318479
1: -0.0162230, 0.0302493, -0.0138386, 0.0261036, -0.0423267, 0.0440879
2: -0.0440542, 0.0211721, -0.0399446, 0.0177537, -0.0618079, 0.0611167
3: -0.0290382, 0.0379892, -0.0261081, 0.0320604, -0.0610986, 0.0640973
4: -0.0557671, 0.0260544, -0.0496877, 0.0220269, -0.0777940, 0.0757421

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546628, upper bound: 0.0536117
time: 0.31 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551628
time: 0.34 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0162888, 0.0169795, -0.0155761, 0.0172236, -0.0335123, 0.0325556
1: -0.0162230, 0.0302493, -0.0150744, 0.0278532, -0.0440762, 0.0453238
2: -0.0440542, 0.0211721, -0.0393254, 0.0200033, -0.0640575, 0.0604976
3: -0.0290382, 0.0379892, -0.0271086, 0.0338280, -0.0628662, 0.0650978
4: -0.0557671, 0.0260544, -0.0510819, 0.0224701, -0.0782371, 0.0771363

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546628, upper bound: 0.0536641
time: 0.34 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551628
time: 0.35 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0156158, 0.0162020, -0.0350453, 0.0362561, -0.0518719, 0.0512473
1: -0.0152479, 0.0284704, -0.0464443, 0.0840745, -0.0993225, 0.0749147
2: -0.0426390, 0.0196745, -0.0838097, 0.0569665, -0.0996055, 0.1034842
3: -0.0277877, 0.0356470, -0.0660995, 0.1159469, -0.1437346, 0.1017464
4: -0.0538614, 0.0244009, -0.1109359, 0.0650300, -0.1188915, 0.1353368

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544499, upper bound: 0.0539956
time: 0.33 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551507, upper bound: 0.0551723
time: 0.33 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0184586, 0.0188769, -0.0466897, 0.0495283
1: -0.0328676, 0.0711690, -0.0188543, 0.0353830, -0.0682506, 0.0900232
2: -0.0700813, 0.0509419, -0.0475542, 0.0247596, -0.0948409, 0.0984961
3: -0.0479890, 0.0927243, -0.0314303, 0.0448089, -0.0927979, 0.1241546
4: -0.0941853, 0.0570478, -0.0593599, 0.0296691, -0.1238543, 0.1164077

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
time: 0.32 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0246860, 0.0260601, -0.0539053, 0.0557837
1: -0.0329159, 0.0712517, -0.0281324, 0.0511528, -0.0840687, 0.0993841
2: -0.0701464, 0.0509891, -0.0582470, 0.0325268, -0.1026731, 0.1092361
3: -0.0480456, 0.0928499, -0.0431673, 0.0669746, -0.1150203, 0.1360172
4: -0.0942536, 0.0571052, -0.0731876, 0.0366854, -0.1309390, 0.1302929

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550812, upper bound: 0.0551371
time: 0.32 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0551720
time: 0.32 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0278452, 0.0310977, -0.0168276, 0.0182708, -0.0461160, 0.0479254
1: -0.0329159, 0.0712517, -0.0175856, 0.0294116, -0.0623275, 0.0888373
2: -0.0701464, 0.0509891, -0.0416592, 0.0218325, -0.0919789, 0.0926484
3: -0.0480456, 0.0928499, -0.0302698, 0.0362866, -0.0843323, 0.1231197
4: -0.0942536, 0.0571052, -0.0531968, 0.0243988, -0.1186524, 0.1103020

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550812, upper bound: 0.0551531
time: 0.32 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552237
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0162253, 0.0168772, -0.0408791, 0.0435025
1: -0.0260409, 0.0594983, -0.0161202, 0.0299492, -0.0559901, 0.0756185
2: -0.0639335, 0.0455851, -0.0438509, 0.0209460, -0.0848795, 0.0894361
3: -0.0402265, 0.0769224, -0.0289131, 0.0375933, -0.0778198, 0.1058355
4: -0.0868411, 0.0507081, -0.0554294, 0.0258221, -0.1126631, 0.1061376

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546765, upper bound: 0.0536285
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549885, upper bound: 0.0551520
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0170802, 0.0188442, -0.0428461, 0.0443574
1: -0.0260409, 0.0594983, -0.0173426, 0.0322402, -0.0582811, 0.0768409
2: -0.0639335, 0.0455851, -0.0431599, 0.0234980, -0.0874315, 0.0887451
3: -0.0402265, 0.0769224, -0.0300243, 0.0405421, -0.0807686, 0.1069468
4: -0.0868411, 0.0507081, -0.0573909, 0.0264609, -0.1133019, 0.1080991

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546765, upper bound: 0.0536809
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549885, upper bound: 0.0551520
time: 0.34 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0372033, 0.0399808, -0.0639827, 0.0644804
1: -0.0260409, 0.0594983, -0.0513292, 0.0966234, -0.1226644, 0.1108275
2: -0.0639335, 0.0455851, -0.0886574, 0.0619009, -0.1258344, 0.1342425
3: -0.0402265, 0.0769224, -0.0712840, 0.1331640, -0.1733906, 0.1482064
4: -0.0868411, 0.0507081, -0.1212188, 0.0697227, -0.1565637, 0.1719269

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547097, upper bound: 0.0534911
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550695, upper bound: 0.0551619
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0240019, 0.0272771, -0.0220613, 0.0258578, -0.0498596, 0.0493385
1: -0.0260409, 0.0594983, -0.0240982, 0.0547428, -0.0807838, 0.0835965
2: -0.0639335, 0.0455851, -0.0583525, 0.0406262, -0.1045597, 0.1039376
3: -0.0402265, 0.0769224, -0.0361878, 0.0701844, -0.1104110, 0.1131102
4: -0.0868411, 0.0507081, -0.0826751, 0.0436150, -0.1304561, 0.1333833

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551116, upper bound: 0.0550157
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551116, upper bound: 0.0550157
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0244671, 0.0285040, -0.0173009, 0.0182213, -0.0426884, 0.0458049
1: -0.0291369, 0.0633202, -0.0177054, 0.0341504, -0.0632873, 0.0810256
2: -0.0632677, 0.0448866, -0.0465042, 0.0232183, -0.0864860, 0.0913908
3: -0.0417759, 0.0822587, -0.0309278, 0.0429934, -0.0847693, 0.1131866
4: -0.0902513, 0.0482991, -0.0592682, 0.0284133, -0.1186646, 0.1075673

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536804, upper bound: 0.0547145
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552055, upper bound: 0.0550711
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0244671, 0.0285040, -0.0191889, 0.0214792, -0.0459462, 0.0476930
1: -0.0291369, 0.0633202, -0.0218908, 0.0408736, -0.0700105, 0.0852110
2: -0.0632677, 0.0448866, -0.0481801, 0.0277699, -0.0910377, 0.0930667
3: -0.0417759, 0.0822587, -0.0356535, 0.0524579, -0.0942338, 0.1179123
4: -0.0902513, 0.0482991, -0.0651084, 0.0310635, -0.1213148, 0.1134076

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 38

Time for candidate selection: 4.88 seconds

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536435, upper bound: 0.0544398
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549123, upper bound: 0.0549707
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0246860, 0.0260601, -0.0380102, 0.0402471, -0.0649330, 0.0640703
1: -0.0281324, 0.0511528, -0.0539456, 0.0992521, -0.1273845, 0.1050984
2: -0.0582470, 0.0325268, -0.0882765, 0.0612784, -0.1195254, 0.1208032
3: -0.0431673, 0.0669746, -0.0748796, 0.1375122, -0.1806795, 0.1418543
4: -0.0731876, 0.0366854, -0.1215643, 0.0684904, -0.1416780, 0.1582496

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551371, upper bound: 0.0550812
time: 0.32 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550924
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0169882, 0.0184047, -0.0380102, 0.0402471, -0.0572352, 0.0564149
1: -0.0178440, 0.0298011, -0.0539456, 0.0992521, -0.1170961, 0.0837467
2: -0.0420441, 0.0220840, -0.0882765, 0.0612784, -0.1033226, 0.1103605
3: -0.0306214, 0.0369190, -0.0748796, 0.1375122, -0.1681336, 0.1117986
4: -0.0535590, 0.0247414, -0.1215643, 0.0684904, -0.1220494, 0.1463057

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551371, upper bound: 0.0550812
time: 0.37 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550924
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0246860, 0.0260601, -0.0253017, 0.0285668, -0.0532528, 0.0513618
1: -0.0281324, 0.0511528, -0.0282283, 0.0612988, -0.0894312, 0.0793810
2: -0.0582470, 0.0325268, -0.0637410, 0.0433148, -0.1015618, 0.0962677
3: -0.0431673, 0.0669746, -0.0414863, 0.0783977, -0.1215650, 0.1084610
4: -0.0731876, 0.0366854, -0.0863181, 0.0472957, -0.1204834, 0.1230034

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550675, upper bound: 0.0550045
time: 0.33 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551024, upper bound: 0.0550157
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0169882, 0.0184047, -0.0253017, 0.0285668, -0.0455550, 0.0437064
1: -0.0178440, 0.0298011, -0.0282283, 0.0612988, -0.0791429, 0.0580294
2: -0.0420441, 0.0220840, -0.0637410, 0.0433148, -0.0853589, 0.0858250
3: -0.0306214, 0.0369190, -0.0414863, 0.0783977, -0.1090191, 0.0784053
4: -0.0535590, 0.0247414, -0.0863181, 0.0472957, -0.1008547, 0.1110595

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550675, upper bound: 0.0550045
time: 0.36 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551024, upper bound: 0.0550157
time: 0.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.65 seconds
IS_A1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0546628, upper bound: 0.0536117
IS_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551628
IS_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0546628, upper bound: 0.0536641
IS_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551628
IS_A1_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0544499, upper bound: 0.0539956
IS_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551507, upper bound: 0.0551723
IS_A1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
IS_A1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
IS_A1_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550812, upper bound: 0.0551371
IS_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0551720
IS_A1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550812, upper bound: 0.0551531
IS_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550924, upper bound: 0.0552237
IS_A1_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0546765, upper bound: 0.0536285
IS_A1_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0549885, upper bound: 0.0551520
IS_A1_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0546765, upper bound: 0.0536809
IS_A1_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0549885, upper bound: 0.0551520
IS_A1_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0547097, upper bound: 0.0534911
IS_A1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550695, upper bound: 0.0551619
IS_A1_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551116, upper bound: 0.0550157
IS_A1_A2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551116, upper bound: 0.0550157
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0536804, upper bound: 0.0547145
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0552055, upper bound: 0.0550711
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0536435, upper bound: 0.0544398
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0549123, upper bound: 0.0549707
IS_A2_B2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551371, upper bound: 0.0550812
IS_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550924
IS_A2_B2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551371, upper bound: 0.0550812
IS_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550924
IS_A2_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550675, upper bound: 0.0550045
IS_A2_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551024, upper bound: 0.0550157
IS_A2_B2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0550675, upper bound: 0.0550045
IS_A2_B2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 0, lower bound: -0.0551024, upper bound: 0.0550157

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0156158, 0.0162020, -0.0148684, 0.0150387, -0.0306545, 0.0310704
1: -0.0152479, 0.0284704, -0.0138386, 0.0261036, -0.0413515, 0.0423089
2: -0.0426390, 0.0196745, -0.0399446, 0.0177537, -0.0603927, 0.0596191
3: -0.0277877, 0.0356470, -0.0261081, 0.0320604, -0.0598482, 0.0617551
4: -0.0538614, 0.0244009, -0.0496877, 0.0220269, -0.0758884, 0.0740886

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551342, upper bound: 0.0551850
time: 0.38 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551342, upper bound: 0.0551342
time: 0.37 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0156158, 0.0162020, -0.0155761, 0.0172236, -0.0328394, 0.0317781
1: -0.0152479, 0.0284704, -0.0150744, 0.0278532, -0.0431011, 0.0435448
2: -0.0426390, 0.0196745, -0.0393254, 0.0200033, -0.0626423, 0.0589999
3: -0.0277877, 0.0356470, -0.0271086, 0.0338280, -0.0616158, 0.0627555
4: -0.0538614, 0.0244009, -0.0510819, 0.0224701, -0.0763315, 0.0754828

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543252, upper bound: 0.0535479
time: 0.39 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551628
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0155749, 0.0161597, -0.0344802, 0.0357422, -0.0513172, 0.0506399
1: -0.0151215, 0.0283850, -0.0450897, 0.0822268, -0.0973483, 0.0734746
2: -0.0425486, 0.0195940, -0.0828358, 0.0562035, -0.0987521, 0.1024298
3: -0.0276105, 0.0355082, -0.0643364, 0.1131167, -0.1407271, 0.0998446
4: -0.0537372, 0.0243104, -0.1095820, 0.0641248, -0.1178620, 0.1338924

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0545708
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551507, upper bound: 0.0551723
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0165931, 0.0174451, -0.0452578, 0.0476628
1: -0.0328676, 0.0711690, -0.0165549, 0.0319198, -0.0647874, 0.0877238
2: -0.0700813, 0.0509419, -0.0450035, 0.0218341, -0.0919154, 0.0959454
3: -0.0479890, 0.0927243, -0.0294085, 0.0398671, -0.0878561, 0.1221327
4: -0.0941853, 0.0570478, -0.0573436, 0.0268115, -0.1209968, 0.1143913

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552487
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0270240, 0.0299775, -0.0577902, 0.0580937
1: -0.0328676, 0.0711690, -0.0311674, 0.0700214, -0.1028890, 0.1023364
2: -0.0700813, 0.0509419, -0.0702907, 0.0497460, -0.1198273, 0.1212326
3: -0.0479890, 0.0927243, -0.0454295, 0.0910918, -0.1390807, 0.1381537
4: -0.0941853, 0.0570478, -0.0959161, 0.0557444, -0.1499297, 0.1529638

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552487
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0237873, 0.0251316, -0.0529443, 0.0548570
1: -0.0328676, 0.0711690, -0.0267833, 0.0485035, -0.0813711, 0.0979522
2: -0.0700813, 0.0509419, -0.0562592, 0.0312938, -0.1013751, 0.1072011
3: -0.0479890, 0.0927243, -0.0414223, 0.0631953, -0.1111842, 0.1341466
4: -0.0941853, 0.0570478, -0.0707915, 0.0350288, -0.1292141, 0.1278392

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0551720
time: 0.37 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0551720
time: 0.38 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0161609, 0.0176300, -0.0454427, 0.0472306
1: -0.0328676, 0.0711690, -0.0166334, 0.0281600, -0.0610276, 0.0878024
2: -0.0700813, 0.0509419, -0.0401816, 0.0208304, -0.0909117, 0.0911235
3: -0.0479890, 0.0927243, -0.0290819, 0.0343263, -0.0823153, 0.1218062
4: -0.0941853, 0.0570478, -0.0514314, 0.0231952, -0.1173805, 0.1084791

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549834, upper bound: 0.0552237
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549834, upper bound: 0.0552237
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0234873, 0.0266846, -0.0372033, 0.0399808, -0.0634682, 0.0638879
1: -0.0250003, 0.0577051, -0.0513292, 0.0966234, -0.1216238, 0.1090343
2: -0.0628079, 0.0445847, -0.0886574, 0.0619009, -0.1247088, 0.1332421
3: -0.0388779, 0.0740361, -0.0712840, 0.1331640, -0.1720419, 0.1453201
4: -0.0853493, 0.0495821, -0.1212188, 0.0697227, -0.1550720, 0.1708009

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0550109
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551774, upper bound: 0.0551774
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0244671, 0.0285040, -0.0166375, 0.0174424, -0.0419095, 0.0451415
1: -0.0291369, 0.0633202, -0.0167646, 0.0319985, -0.0611355, 0.0800848
2: -0.0632677, 0.0448866, -0.0451248, 0.0217862, -0.0850539, 0.0900114
3: -0.0417759, 0.0822587, -0.0297192, 0.0399729, -0.0817488, 0.1119779
4: -0.0902513, 0.0482991, -0.0574910, 0.0268244, -0.1170757, 0.1057902

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0551697
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0551507
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0237873, 0.0251316, -0.0379789, 0.0402195, -0.0640068, 0.0631104
1: -0.0267833, 0.0485035, -0.0538915, 0.0991745, -0.1259577, 0.1023950
2: -0.0562592, 0.0312938, -0.0882134, 0.0612372, -0.1174964, 0.1195073
3: -0.0414223, 0.0631953, -0.0748124, 0.1373984, -0.1788207, 0.1380076
4: -0.0707915, 0.0350288, -0.1214939, 0.0684392, -0.1392306, 0.1565227

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550701
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0551791
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0162622, 0.0177198, -0.0379789, 0.0402195, -0.0564817, 0.0556986
1: -0.0168090, 0.0283452, -0.0538915, 0.0991745, -0.1159834, 0.0822368
2: -0.0404315, 0.0209995, -0.0882134, 0.0612372, -0.1016687, 0.1092129
3: -0.0293162, 0.0346009, -0.0748124, 0.1373984, -0.1667146, 0.1094133
4: -0.0516704, 0.0234195, -0.1214939, 0.0684392, -0.1201095, 0.1449134

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552237, upper bound: 0.0549834
time: 0.38 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552237, upper bound: 0.0550924
time: 0.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.22 seconds
IS_A1_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551342, upper bound: 0.0551850
IS_A1_A1_B2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551342, upper bound: 0.0551342
IS_A1_A1_B2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0543252, upper bound: 0.0535479
IS_A1_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550324, upper bound: 0.0551628
IS_A1_A1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0545708
IS_A1_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551507, upper bound: 0.0551723
IS_A1_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552487
IS_A1_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
IS_A1_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552487
IS_A1_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0552610
IS_A1_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0551720
IS_A1_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0551720
IS_A1_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0549834, upper bound: 0.0552237
IS_A1_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0549834, upper bound: 0.0552237
IS_A1_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0550109
IS_A1_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551774, upper bound: 0.0551774
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0551697
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0551507
IS_A2_B2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550701
IS_A2_B2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0551791
IS_A2_B2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0552237, upper bound: 0.0549834
IS_A2_B2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.22
Output dim: 0, lower bound: -0.0552237, upper bound: 0.0550924

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0174951, 0.0171407, -0.0148684, 0.0150387, -0.0325337, 0.0320091
1: -0.0167917, 0.0279771, -0.0138386, 0.0261036, -0.0428953, 0.0418156
2: -0.0433687, 0.0208287, -0.0399446, 0.0177537, -0.0611224, 0.0607733
3: -0.0293001, 0.0343063, -0.0261081, 0.0320604, -0.0613606, 0.0604144
4: -0.0519679, 0.0254249, -0.0496877, 0.0220269, -0.0739948, 0.0751125

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545536, upper bound: 0.0548758
time: 0.39 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551342, upper bound: 0.0551850
time: 0.38 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0152079, 0.0158311, -0.0155397, 0.0171908, -0.0323987, 0.0313708
1: -0.0146773, 0.0277137, -0.0150281, 0.0277992, -0.0424765, 0.0427418
2: -0.0417468, 0.0188577, -0.0392561, 0.0199481, -0.0616948, 0.0581137
3: -0.0270949, 0.0344963, -0.0270498, 0.0337503, -0.0608451, 0.0615461
4: -0.0527696, 0.0234246, -0.0510092, 0.0223983, -0.0751679, 0.0744338

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 7

Time for candidate selection: 4.20 seconds

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544460, upper bound: 0.0538499
time: 0.36 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548099, upper bound: 0.0549332
time: 0.40 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0151713, 0.0157911, -0.0344533, 0.0357183, -0.0508896, 0.0502444
1: -0.0145505, 0.0276366, -0.0450441, 0.0821555, -0.0967060, 0.0726807
2: -0.0416646, 0.0187884, -0.0827833, 0.0561613, -0.0978258, 0.1015718
3: -0.0269150, 0.0343694, -0.0642810, 0.1130099, -0.1399249, 0.0986504
4: -0.0526556, 0.0233465, -0.1095160, 0.0640747, -0.1167303, 0.1328625

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551238, upper bound: 0.0551723
time: 0.39 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551238, upper bound: 0.0549606
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0183251, 0.0186145, -0.0464272, 0.0493948
1: -0.0328676, 0.0711690, -0.0180574, 0.0323794, -0.0652470, 0.0892264
2: -0.0700813, 0.0509419, -0.0455349, 0.0230801, -0.0931614, 0.0964768
3: -0.0479890, 0.0927243, -0.0309879, 0.0395181, -0.0875071, 0.1237122
4: -0.0941853, 0.0570478, -0.0543937, 0.0276460, -0.1218313, 0.1114415

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0552568
time: 0.36 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551230
time: 0.41 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0144423, 0.0144348, -0.0422475, 0.0455120
1: -0.0328676, 0.0711690, -0.0134045, 0.0251642, -0.0580318, 0.0845734
2: -0.0700813, 0.0509419, -0.0389097, 0.0167449, -0.0868262, 0.0898516
3: -0.0479890, 0.0927243, -0.0256239, 0.0306482, -0.0786372, 0.1183482
4: -0.0941853, 0.0570478, -0.0483217, 0.0208501, -0.1150354, 0.1053695

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0552610
time: 0.40 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551230
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0272136, 0.0296514, -0.0574641, 0.0582833
1: -0.0328676, 0.0711690, -0.0316533, 0.0673123, -0.1001799, 0.1028223
2: -0.0700813, 0.0509419, -0.0682733, 0.0478253, -0.1179066, 0.1192152
3: -0.0479890, 0.0927243, -0.0464988, 0.0873095, -0.1352985, 0.1392231
4: -0.0941853, 0.0570478, -0.0909609, 0.0538906, -0.1480758, 0.1480087

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548814, upper bound: 0.0552487
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548814, upper bound: 0.0551230
time: 0.39 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0278127, 0.0310697, -0.0234784, 0.0260838, -0.0538965, 0.0545481
1: -0.0328676, 0.0711690, -0.0249705, 0.0563349, -0.0892025, 0.0961395
2: -0.0700813, 0.0509419, -0.0624155, 0.0433351, -0.1134164, 0.1133574
3: -0.0479890, 0.0927243, -0.0388811, 0.0726890, -0.1206779, 0.1316053
4: -0.0941853, 0.0570478, -0.0840991, 0.0484297, -0.1426150, 0.1411469

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548814, upper bound: 0.0552610
time: 0.46 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548814, upper bound: 0.0551114
time: 0.37 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0278679, 0.0321473, -0.0237873, 0.0251316, -0.0529994, 0.0559346
1: -0.0329683, 0.0750322, -0.0267833, 0.0485035, -0.0814718, 0.1018154
2: -0.0702526, 0.0525257, -0.0562592, 0.0312938, -0.1015464, 0.1087849
3: -0.0474031, 0.0969662, -0.0414223, 0.0631953, -0.1105984, 0.1383886
4: -0.0962531, 0.0578964, -0.0707915, 0.0350288, -0.1312818, 0.1286879

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38

Time for candidate selection: 4.12 seconds

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550701, upper bound: 0.0551720
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547030, upper bound: 0.0550307
time: 0.38 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546514, upper bound: 0.0545048
time: 0.35 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273508, 0.0306702, -0.0237873, 0.0251316, -0.0524824, 0.0544575
1: -0.0321755, 0.0699852, -0.0267833, 0.0485035, -0.0806790, 0.0967685
2: -0.0691589, 0.0502736, -0.0562592, 0.0312938, -0.1004527, 0.1065328
3: -0.0471808, 0.0909255, -0.0414223, 0.0631953, -0.1103761, 0.1323479
4: -0.0932161, 0.0562352, -0.0707915, 0.0350288, -0.1282449, 0.1270267

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.14 seconds

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 10

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547030, upper bound: 0.0550126
time: 0.42 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546514, upper bound: 0.0544830
time: 0.40 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0278679, 0.0321473, -0.0161609, 0.0176300, -0.0454978, 0.0483082
1: -0.0329683, 0.0750322, -0.0166334, 0.0281600, -0.0611283, 0.0916656
2: -0.0702526, 0.0525257, -0.0401816, 0.0208304, -0.0910829, 0.0927073
3: -0.0474031, 0.0969662, -0.0290819, 0.0343263, -0.0817294, 0.1260481
4: -0.0962531, 0.0578964, -0.0514314, 0.0231952, -0.1194482, 0.1093278

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.21 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541466, upper bound: 0.0540784
time: 0.39 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547940, upper bound: 0.0550943
time: 0.42 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0273508, 0.0306702, -0.0161609, 0.0176300, -0.0449808, 0.0468311
1: -0.0321755, 0.0699852, -0.0166334, 0.0281600, -0.0603355, 0.0866186
2: -0.0691589, 0.0502736, -0.0401816, 0.0208304, -0.0899892, 0.0904552
3: -0.0471808, 0.0909255, -0.0290819, 0.0343263, -0.0815071, 0.1200074
4: -0.0932161, 0.0562352, -0.0514314, 0.0231952, -0.1164113, 0.1076666

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 38

Time for candidate selection: 4.31 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541466, upper bound: 0.0541584
time: 0.34 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547940, upper bound: 0.0550159
time: 0.36 seconds

## BFS IS instance: IS_A1_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0230536, 0.0262401, -0.0371749, 0.0399542, -0.0630078, 0.0634151
1: -0.0242822, 0.0563962, -0.0512880, 0.0965437, -0.1208260, 0.1076842
2: -0.0619228, 0.0438159, -0.0885991, 0.0618547, -0.1237775, 0.1324149
3: -0.0380059, 0.0721582, -0.0712343, 0.1330481, -0.1710541, 0.1433925
4: -0.0842291, 0.0486820, -0.1211443, 0.0696681, -0.1538972, 0.1698263

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551774, upper bound: 0.0550682
time: 0.35 seconds

## Relational analysis of IS_A1_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551774, upper bound: 0.0550682
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0257471, 0.0290832, -0.0163582, 0.0171201, -0.0428671, 0.0454414
1: -0.0298784, 0.0626745, -0.0160783, 0.0308858, -0.0607642, 0.0787528
2: -0.0645730, 0.0442843, -0.0445815, 0.0213400, -0.0859131, 0.0888658
3: -0.0435096, 0.0804226, -0.0287350, 0.0384864, -0.0819959, 0.1091576
4: -0.0875694, 0.0483158, -0.0567655, 0.0263052, -0.1138746, 0.1050813

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543961, upper bound: 0.0549982
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0551681
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0199965, 0.0235820, -0.0155387, 0.0161072, -0.0361037, 0.0391208
1: -0.0205243, 0.0465149, -0.0150583, 0.0282544, -0.0487787, 0.0615732
2: -0.0532895, 0.0369669, -0.0424238, 0.0194556, -0.0727451, 0.0793907
3: -0.0326785, 0.0591032, -0.0275335, 0.0353154, -0.0679939, 0.0866368
4: -0.0750140, 0.0393477, -0.0535276, 0.0241639, -0.0991779, 0.0928753

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545708, upper bound: 0.0549097
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0551507
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0237873, 0.0251316, -0.0390759, 0.0418176, -0.0656049, 0.0642075
1: -0.0267833, 0.0485035, -0.0566049, 0.1040098, -0.1307931, 0.1051084
2: -0.0562592, 0.0312938, -0.0898096, 0.0635595, -0.1198187, 0.1211035
3: -0.0414223, 0.0631953, -0.0783719, 0.1435497, -0.1849720, 0.1415672
4: -0.0707915, 0.0350288, -0.1246866, 0.0701820, -0.1409735, 0.1597154

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38

Time for candidate selection: 4.22 seconds

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551720, upper bound: 0.0550701
time: 0.38 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550307, upper bound: 0.0547030
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545048, upper bound: 0.0546514
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0237873, 0.0251316, -0.0375685, 0.0398326, -0.0636199, 0.0627001
1: -0.0267833, 0.0485035, -0.0531399, 0.0980796, -0.1248629, 0.1016434
2: -0.0562592, 0.0312938, -0.0874201, 0.0606558, -0.1169151, 0.1187140
3: -0.0414223, 0.0631953, -0.0738551, 0.1358038, -0.1772261, 0.1370503
4: -0.0707915, 0.0350288, -0.1205179, 0.0677188, -0.1385103, 0.1555467

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 38

Time for candidate selection: 4.11 seconds

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 10

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550307, upper bound: 0.0547030
time: 0.34 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545048, upper bound: 0.0546514
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0162622, 0.0177198, -0.0390759, 0.0418176, -0.0580798, 0.0567957
1: -0.0168090, 0.0283452, -0.0566049, 0.1040098, -0.1208188, 0.0849501
2: -0.0404315, 0.0209995, -0.0898096, 0.0635595, -0.1039910, 0.1108091
3: -0.0293162, 0.0346009, -0.0783719, 0.1435497, -0.1728658, 0.1129728
4: -0.0516704, 0.0234195, -0.1246866, 0.0701820, -0.1218524, 0.1481061

Time for backsubstitution: 1.93 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0036636, high=0.0282057, mid=0.0282057, abs_max=0.058847926557064056
rel_dist={0: [-0.055451745298252274, 0.05545174529825231]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.003663635035536572
execution time: 1147.61 seconds
