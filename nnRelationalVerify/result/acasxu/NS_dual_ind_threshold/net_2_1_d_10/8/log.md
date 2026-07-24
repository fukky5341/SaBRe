## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 77.93799558274


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.9465866, 50.7789307, -41.9465866, 50.7789307, -92.7255173, 92.7255173)
1: (-32.5055008, 40.2750320, -32.5055008, 40.2750320, -72.7805176, 72.7805176)
2: (-28.2902870, 40.3181725, -28.2902870, 40.3181725, -68.6084595, 68.6084595)
3: (-39.0206070, 48.2061768, -39.0206070, 48.2061768, -87.2267838, 87.2267838)
4: (-36.7614212, 53.8471222, -36.7614212, 53.8471222, -90.6085358, 90.6085358)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 2.16 = 3.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -77.9535863, upper bound: 77.9535863

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9403407, upper bound: 77.9532077
time: 0.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
time: 1.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 3, lower bound: -77.9403407, upper bound: 77.9532077
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -36.9003334, 44.3449478, -41.9465866, 50.7789307, -87.6792603, 86.2915344
1: -28.5048008, 35.1618042, -32.5055008, 40.2750320, -68.7798309, 67.6672897
2: -24.8153992, 35.1510353, -28.2902870, 40.3181725, -65.1335754, 63.4413223
3: -34.2013397, 42.0720787, -39.0206070, 48.2061768, -82.4075165, 81.0926819
4: -32.1969414, 46.9156647, -36.7614212, 53.8471222, -86.0440674, 83.6770782

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
time: 0.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
time: 1.07 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -49.1125984, 59.5777740, -41.9465866, 50.7789307, -99.8915253, 101.5243607
1: -38.0327377, 47.2213669, -32.5055008, 40.2750320, -78.3077698, 79.7268524
2: -33.1714363, 47.3327522, -28.2902870, 40.3181725, -73.4896088, 75.6230392
3: -46.0302086, 56.3861580, -39.0206070, 48.2061768, -94.2363892, 95.4067688
4: -43.0803299, 63.3247147, -36.7614212, 53.8471222, -96.9274521, 100.0861206

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
time: 0.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
time: 0.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.32 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.32
Output dim: 3, lower bound: -77.9400895, upper bound: 77.9400895

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -36.9003334, 44.3449478, -36.9003334, 44.3449478, -81.2452621, 81.2452621
1: -28.5048008, 35.1618042, -28.5048008, 35.1618042, -63.6666031, 63.6666031
2: -24.8153992, 35.1510353, -24.8153992, 35.1510353, -59.9664345, 59.9664345
3: -34.2013397, 42.0720787, -34.2013397, 42.0720787, -76.2734222, 76.2734222
4: -32.1969414, 46.9156647, -32.1969414, 46.9156647, -79.1126099, 79.1126099

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9402906, upper bound: 77.9529037
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399275, upper bound: 77.9408101
time: 0.65 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -36.9003334, 44.3449478, -49.1125984, 59.5777740, -96.4780960, 93.4575424
1: -28.5048008, 35.1618042, -38.0327377, 47.2213669, -75.7261658, 73.1945190
2: -24.8153992, 35.1510353, -33.1714363, 47.3327522, -72.1481476, 68.3224716
3: -34.2013397, 42.0720787, -46.0302086, 56.3861580, -90.5874939, 88.1022873
4: -32.1969414, 46.9156647, -43.0803299, 63.3247147, -95.5216522, 89.9959946

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9402906, upper bound: 77.9529366
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399275, upper bound: 77.9408430
time: 0.64 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -49.1125984, 59.5777740, -36.9003334, 44.3449478, -93.4575424, 96.4780960
1: -38.0327377, 47.2213669, -28.5048008, 35.1618042, -73.1945190, 75.7261658
2: -33.1714363, 47.3327522, -24.8153992, 35.1510353, -68.3224716, 72.1481476
3: -46.0302086, 56.3861580, -34.2013397, 42.0720787, -88.1022873, 90.5874939
4: -43.0803299, 63.3247147, -32.1969414, 46.9156647, -89.9959946, 95.5216522

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400831, upper bound: 77.9398699
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399275, upper bound: 77.9399275
time: 0.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -49.1125984, 59.5777740, -49.1125984, 59.5777740, -108.6903687, 108.6903687
1: -38.0327377, 47.2213669, -38.0327377, 47.2213669, -85.2540894, 85.2540894
2: -33.1714363, 47.3327522, -33.1714363, 47.3327522, -80.5041885, 80.5041885
3: -46.0302086, 56.3861580, -46.0302086, 56.3861580, -102.4163666, 102.4163666
4: -43.0803299, 63.3247147, -43.0803299, 63.3247147, -106.4050446, 106.4050446

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400831, upper bound: 77.9398699
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399604, upper bound: 77.9399275
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.96 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9402906, upper bound: 77.9529037
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9399275, upper bound: 77.9408101
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9402906, upper bound: 77.9529366
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9399275, upper bound: 77.9408430
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9400831, upper bound: 77.9398699
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9399275, upper bound: 77.9399275
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9400831, upper bound: 77.9398699
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 3, lower bound: -77.9399604, upper bound: 77.9399275

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -35.6116638, 42.6447182, -36.9003334, 44.3449478, -79.9566116, 79.5450363
1: -27.4477100, 33.7920532, -28.5048008, 35.1618042, -62.6095123, 62.2968521
2: -23.8948441, 33.7621422, -24.8153992, 35.1510353, -59.0458794, 58.5775414
3: -32.9314537, 40.4353333, -34.2013397, 42.0720787, -75.0035172, 74.6366730
4: -30.9905396, 45.0463181, -32.1969414, 46.9156647, -77.9062042, 77.2432480

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -38.8135300, 46.2989769, -36.9003334, 44.3449478, -83.1584473, 83.1992950
1: -29.8790951, 36.7157326, -28.5048008, 35.1618042, -65.0408936, 65.2205353
2: -26.0351105, 36.6458130, -24.8153992, 35.1510353, -61.1861458, 61.4612122
3: -35.8771858, 43.9032440, -34.2013397, 42.0720787, -77.9492645, 78.1045837
4: -33.7433815, 48.9063873, -32.1969414, 46.9156647, -80.6590424, 81.1033173

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -35.6116638, 42.6447182, -49.1125984, 59.5777740, -95.1894379, 91.7573090
1: -27.4477100, 33.7920532, -38.0327377, 47.2213669, -74.6690750, 71.8247910
2: -23.8948441, 33.7621422, -33.1714363, 47.3327522, -71.2276001, 66.9335785
3: -32.9314537, 40.4353333, -46.0302086, 56.3861580, -89.3175888, 86.4655457
4: -30.9905396, 45.0463181, -43.0803299, 63.3247147, -94.3152542, 88.1266327

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -38.8135300, 46.2989769, -49.1125984, 59.5777740, -98.3912811, 95.4115677
1: -29.8790951, 36.7157326, -38.0327377, 47.2213669, -77.1004562, 74.7484589
2: -26.0351105, 36.6458130, -33.1714363, 47.3327522, -73.3678589, 69.8172455
3: -35.8771858, 43.9032440, -46.0302086, 56.3861580, -92.2633438, 89.9334564
4: -33.7433815, 48.9063873, -43.0803299, 63.3247147, -97.0681000, 91.9867096

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -47.6672821, 57.7376595, -36.9003334, 44.3449478, -92.0122299, 94.6379776
1: -36.8651505, 45.7337990, -28.5048008, 35.1618042, -72.0269470, 74.2386017
2: -32.1516266, 45.8342133, -24.8153992, 35.1510353, -67.3026581, 70.6496124
3: -44.6289749, 54.6052475, -34.2013397, 42.0720787, -86.7010498, 88.8065872
4: -41.7495308, 61.3076630, -32.1969414, 46.9156647, -88.6651917, 93.5046005

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408430, upper bound: 77.9398699
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408430, upper bound: 77.9398699
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -49.4725609, 59.7433586, -36.9003334, 44.3449478, -93.8175049, 96.6436920
1: -38.2433777, 47.3670807, -28.5048008, 35.1618042, -73.4051743, 75.8718796
2: -33.3768120, 47.4269867, -24.8153992, 35.1510353, -68.5278473, 72.2423706
3: -46.3069305, 56.5333519, -34.2013397, 42.0720787, -88.3790131, 90.7346954
4: -43.3185921, 63.4404526, -32.1969414, 46.9156647, -90.2342529, 95.6373901

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9399275
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9399275
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -47.6672821, 57.7376595, -49.1125984, 59.5777740, -107.2450562, 106.8502502
1: -36.8651505, 45.7337990, -38.0327377, 47.2213669, -84.0865097, 83.7665253
2: -32.1516266, 45.8342133, -33.1714363, 47.3327522, -79.4843750, 79.0056458
3: -44.6289749, 54.6052475, -46.0302086, 56.3861580, -101.0151367, 100.6354523
4: -41.7495308, 61.3076630, -43.0803299, 63.3247147, -105.0742493, 104.3879776

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399028, upper bound: 77.9398699
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399028, upper bound: 77.9398699
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -49.4725609, 59.7433586, -49.1125984, 59.5777740, -109.0503387, 108.8559570
1: -38.2433777, 47.3670807, -38.0327377, 47.2213669, -85.4647293, 85.3998184
2: -33.3768120, 47.4269867, -33.1714363, 47.3327522, -80.7095642, 80.5984192
3: -46.3069305, 56.5333519, -46.0302086, 56.3861580, -102.6930847, 102.5635605
4: -43.3185921, 63.4404526, -43.0803299, 63.3247147, -106.6433105, 106.5207825

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9399275
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9399275
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.76 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9408101
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9408430
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408430, upper bound: 77.9398699
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408430, upper bound: 77.9398699
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9399275
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9408101, upper bound: 77.9399275
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9399028, upper bound: 77.9398699
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9399028, upper bound: 77.9398699
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9399275
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 3, lower bound: -77.9398699, upper bound: 77.9399275

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.6116638, 42.6447182, -35.6116638, 42.6447182, -78.2563782, 78.2563782
1: -27.4477100, 33.7920532, -27.4477100, 33.7920532, -61.2397614, 61.2397614
2: -23.8948441, 33.7621422, -23.8948441, 33.7621422, -57.6569862, 57.6569862
3: -32.9314537, 40.4353333, -32.9314537, 40.4353333, -73.3667831, 73.3667831
4: -30.9905396, 45.0463181, -30.9905396, 45.0463181, -76.0368500, 76.0368500

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9405898, upper bound: 77.9400626
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9409706, upper bound: 77.9526772
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.6116638, 42.6447182, -38.8135300, 46.2989769, -81.9106369, 81.4582214
1: -27.4477100, 33.7920532, -29.8790951, 36.7157326, -64.1634369, 63.6711502
2: -23.8948441, 33.7621422, -26.0351105, 36.6458130, -60.5406570, 59.7972527
3: -32.9314537, 40.4353333, -35.8771858, 43.9032440, -76.8346710, 76.3125153
4: -30.9905396, 45.0463181, -33.7433815, 48.9063873, -79.8969193, 78.7896881

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9405898, upper bound: 77.9400626
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9409706, upper bound: 77.9526772
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -38.8135300, 46.2989769, -35.6116638, 42.6447182, -81.4582214, 81.9106369
1: -29.8790951, 36.7157326, -27.4477100, 33.7920532, -63.6711502, 64.1634445
2: -26.0351105, 36.6458130, -23.8948441, 33.7621422, -59.7972527, 60.5406570
3: -35.8771858, 43.9032440, -32.9314537, 40.4353333, -76.3125153, 76.8346710
4: -33.7433815, 48.9063873, -30.9905396, 45.0463181, -78.7896881, 79.8969193

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358544, upper bound: 77.9403662
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -38.8135300, 46.2989769, -38.8135300, 46.2989769, -85.1124802, 85.1124802
1: -29.8790951, 36.7157326, -29.8790951, 36.7157326, -66.5948257, 66.5948257
2: -26.0351105, 36.6458130, -26.0351105, 36.6458130, -62.6809196, 62.6809235
3: -35.8771858, 43.9032440, -35.8771858, 43.9032440, -79.7804260, 79.7804260
4: -33.7433815, 48.9063873, -33.7433815, 48.9063873, -82.6497650, 82.6497650

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358544, upper bound: 77.9403662
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -35.6116638, 42.6447182, -47.6672821, 57.7376595, -93.3493195, 90.3119965
1: -27.4477100, 33.7920532, -36.8651505, 45.7337990, -73.1815109, 70.6572037
2: -23.8948441, 33.7621422, -32.1516266, 45.8342133, -69.7290573, 65.9137650
3: -32.9314537, 40.4353333, -44.6289749, 54.6052475, -87.5366821, 85.0643082
4: -30.9905396, 45.0463181, -41.7495308, 61.3076630, -92.2982025, 86.7958374

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9396792, upper bound: 77.9400961
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400600, upper bound: 77.9527106
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -35.6116638, 42.6447182, -49.4725609, 59.7433586, -95.3550262, 92.1172791
1: -27.4477100, 33.7920532, -38.2433777, 47.3670807, -74.8147888, 72.0354309
2: -23.8948441, 33.7621422, -33.3768120, 47.4269867, -71.3218231, 67.1389542
3: -32.9314537, 40.4353333, -46.3069305, 56.5333519, -89.4647980, 86.7422638
4: -30.9905396, 45.0463181, -43.3185921, 63.4404526, -94.4309921, 88.3649139

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9396792, upper bound: 77.9400961
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400600, upper bound: 77.9527106
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -38.8135300, 46.2989769, -47.6672821, 57.7376595, -96.5511627, 93.9662552
1: -29.8790951, 36.7157326, -36.8651505, 45.7337990, -75.6128845, 73.5808792
2: -26.0351105, 36.6458130, -32.1516266, 45.8342133, -71.8693237, 68.7974243
3: -35.8771858, 43.9032440, -44.6289749, 54.6052475, -90.4824371, 88.5322189
4: -33.7433815, 48.9063873, -41.7495308, 61.3076630, -95.0510330, 90.6559143

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384973, upper bound: 77.9406188
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -38.8135300, 46.2989769, -49.4725609, 59.7433586, -98.5568848, 95.7715378
1: -29.8790951, 36.7157326, -38.2433777, 47.3670807, -77.2461777, 74.9590988
2: -26.0351105, 36.6458130, -33.3768120, 47.4269867, -73.4620895, 70.0226135
3: -35.8771858, 43.9032440, -46.3069305, 56.5333519, -92.4105377, 90.2101746
4: -33.7433815, 48.9063873, -43.3185921, 63.4404526, -97.1838379, 92.2249756

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384973, upper bound: 77.9406188
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -47.6672821, 57.7376595, -35.6116638, 42.6447182, -90.3119965, 93.3493195
1: -36.8651505, 45.7337990, -27.4477100, 33.7920532, -70.6572037, 73.1815109
2: -32.1516266, 45.8342133, -23.8948441, 33.7621422, -65.9137650, 69.7290573
3: -44.6289749, 54.6052475, -32.9314537, 40.4353333, -85.0643082, 87.5366821
4: -41.7495308, 61.3076630, -30.9905396, 45.0463181, -86.7958374, 92.2981949

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9404669, upper bound: 77.9305723
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9407650, upper bound: 77.9396972
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -47.6672821, 57.7376595, -38.8135300, 46.2989769, -93.9662552, 96.5511627
1: -36.8651505, 45.7337990, -29.8790951, 36.7157326, -73.5808792, 75.6128845
2: -32.1516266, 45.8342133, -26.0351105, 36.6458130, -68.7974243, 71.8693237
3: -44.6289749, 54.6052475, -35.8771858, 43.9032440, -88.5322189, 90.4824371
4: -41.7495308, 61.3076630, -33.7433815, 48.9063873, -90.6559143, 95.0510330

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9404669, upper bound: 77.9305723
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9407650, upper bound: 77.9396972
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -49.4725609, 59.7433586, -35.6116638, 42.6447182, -92.1172791, 95.3550262
1: -38.2433777, 47.3670807, -27.4477100, 33.7920532, -72.0354309, 74.8147888
2: -33.3768120, 47.4269867, -23.8948441, 33.7621422, -67.1389542, 71.3218231
3: -46.3069305, 56.5333519, -32.9314537, 40.4353333, -86.7422638, 89.4647980
4: -43.3185921, 63.4404526, -30.9905396, 45.0463181, -88.3649139, 94.4309921

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358541, upper bound: 77.9394788
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358765, upper bound: 77.9382668
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -49.4725609, 59.7433586, -38.8135300, 46.2989769, -95.7715378, 98.5568848
1: -38.2433777, 47.3670807, -29.8790951, 36.7157326, -74.9590988, 77.2461777
2: -33.3768120, 47.4269867, -26.0351105, 36.6458130, -70.0226135, 73.4620895
3: -46.3069305, 56.5333519, -35.8771858, 43.9032440, -90.2101669, 92.4105377
4: -43.3185921, 63.4404526, -33.7433815, 48.9063873, -92.2249756, 97.1838379

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358541, upper bound: 77.9394788
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358765, upper bound: 77.9382668
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -47.6672821, 57.7376595, -47.6672821, 57.7376595, -105.4049377, 105.4049377
1: -36.8651505, 45.7337990, -36.8651505, 45.7337990, -82.5989532, 82.5989532
2: -32.1516266, 45.8342133, -32.1516266, 45.8342133, -77.9858398, 77.9858398
3: -44.6289749, 54.6052475, -44.6289749, 54.6052475, -99.2342224, 99.2342224
4: -41.7495308, 61.3076630, -41.7495308, 61.3076630, -103.0571899, 103.0571823

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395563, upper bound: 77.9305723
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398544, upper bound: 77.9396972
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -47.6672821, 57.7376595, -49.4725609, 59.7433586, -107.4106445, 107.2102203
1: -36.8651505, 45.7337990, -38.2433777, 47.3670807, -84.2322311, 83.9771652
2: -32.1516266, 45.8342133, -33.3768120, 47.4269867, -79.5785980, 79.2110291
3: -44.6289749, 54.6052475, -46.3069305, 56.5333519, -101.1623230, 100.9121780
4: -41.7495308, 61.3076630, -43.3185921, 63.4404526, -105.1899872, 104.6262512

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395563, upper bound: 77.9305723
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398544, upper bound: 77.9396972
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -49.4725609, 59.7433586, -47.6672821, 57.7376595, -107.2102203, 107.4106445
1: -38.2433777, 47.3670807, -36.8651505, 45.7337990, -83.9771652, 84.2322311
2: -33.3768120, 47.4269867, -32.1516266, 45.8342133, -79.2110291, 79.5785980
3: -46.3069305, 56.5333519, -44.6289749, 54.6052475, -100.9121780, 101.1623230
4: -43.3185921, 63.4404526, -41.7495308, 61.3076630, -104.6262512, 105.1899872

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9394788
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.4725609, 59.7433586, -49.4725609, 59.7433586, -109.2159195, 109.2159195
1: -38.2433777, 47.3670807, -38.2433777, 47.3670807, -85.6104584, 85.6104584
2: -33.3768120, 47.4269867, -33.3768120, 47.4269867, -80.8037872, 80.8037872
3: -46.3069305, 56.5333519, -46.3069305, 56.5333519, -102.8402786, 102.8402786
4: -43.3185921, 63.4404526, -43.3185921, 63.4404526, -106.7590485, 106.7590485

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9394788
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835
time: 0.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.36 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9405898, upper bound: 77.9400626
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9409706, upper bound: 77.9526772
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9405898, upper bound: 77.9400626
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9409706, upper bound: 77.9526772
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9358544, upper bound: 77.9403662
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9358544, upper bound: 77.9403662
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9396792, upper bound: 77.9400961
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9400600, upper bound: 77.9527106
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9396792, upper bound: 77.9400961
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9400600, upper bound: 77.9527106
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9384973, upper bound: 77.9406188
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9384973, upper bound: 77.9406188
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9404669, upper bound: 77.9305723
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9407650, upper bound: 77.9396972
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9404669, upper bound: 77.9305723
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9407650, upper bound: 77.9396972
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9358541, upper bound: 77.9394788
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9358765, upper bound: 77.9382668
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9358541, upper bound: 77.9394788
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9358765, upper bound: 77.9382668
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9395563, upper bound: 77.9305723
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9398544, upper bound: 77.9396972
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9395563, upper bound: 77.9305723
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9398544, upper bound: 77.9396972
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9394788
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9394788
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -35.6116638, 42.6447182, -76.3454285, 75.7906036
1: -25.9439163, 31.8255253, -27.4477100, 33.7920532, -59.7359695, 59.2732353
2: -22.5959740, 31.7507057, -23.8948441, 33.7621422, -56.3581161, 55.6455460
3: -31.1403046, 38.0657272, -32.9314537, 40.4353333, -71.5756378, 70.9971466
4: -29.2728157, 42.3466530, -30.9905396, 45.0463181, -74.3191376, 73.3371887

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400446, upper bound: 77.9400446
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9400446, upper bound: 77.9400446
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -35.6116638, 42.6447182, -77.9321899, 77.8278427
1: -27.1880989, 33.4509544, -27.4477100, 33.7920532, -60.9801521, 60.8986664
2: -23.6696396, 33.4155960, -23.8948441, 33.7621422, -57.4317780, 57.3104362
3: -32.6121635, 40.0253601, -32.9314537, 40.4353333, -73.0475006, 72.9567947
4: -30.6917725, 44.5793228, -30.9905396, 45.0463181, -75.7380905, 75.5698624

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9404254, upper bound: 77.9526592
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9404254, upper bound: 77.9530400
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -38.8135300, 46.2989769, -79.9996872, 78.9924622
1: -25.9439163, 31.8255253, -29.8790951, 36.7157326, -62.6596489, 61.7046204
2: -22.5959740, 31.7507057, -26.0351105, 36.6458130, -59.2417870, 57.7858162
3: -31.1403046, 38.0657272, -35.8771858, 43.9032440, -75.0435486, 73.9429092
4: -29.2728157, 42.3466530, -33.7433815, 48.9063873, -78.1791992, 76.0900345

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9403754, upper bound: 77.9381839
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9347685, upper bound: 77.9379267
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -38.8135300, 46.2989769, -81.5864563, 81.0296860
1: -27.1880989, 33.4509544, -29.8790951, 36.7157326, -63.9038200, 63.3300438
2: -23.6696396, 33.4155960, -26.0351105, 36.6458130, -60.3154449, 59.4507065
3: -32.6121635, 40.0253601, -35.8771858, 43.9032440, -76.5153961, 75.9025421
4: -30.6917725, 44.5793228, -33.7433815, 48.9063873, -79.5981598, 78.3227081

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9407917, upper bound: 77.9517370
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9351848, upper bound: 77.9514798
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -35.6116638, 42.6447182, -68.7564316, 66.0714951
1: -19.9750519, 24.0622005, -27.4477100, 33.7920532, -53.7671051, 51.5099106
2: -17.3923473, 24.0895405, -23.8948441, 33.7621422, -51.1544876, 47.9843826
3: -23.9274788, 28.7653179, -32.9314537, 40.4353333, -64.3628082, 61.6967659
4: -22.4313278, 32.1443291, -30.9905396, 45.0463181, -67.4776154, 63.1348686

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381839, upper bound: 77.9403754
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9517370, upper bound: 77.9407917
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -38.8135300, 46.2989769, -72.4106903, 69.2733307
1: -19.9750519, 24.0622005, -29.8790951, 36.7157326, -56.6907845, 53.9412956
2: -17.3923473, 24.0895405, -26.0351105, 36.6458130, -54.0381622, 50.1246490
3: -23.9274788, 28.7653179, -35.8771858, 43.9032440, -67.8307190, 64.6425018
4: -22.4313278, 32.1443291, -33.7433815, 48.9063873, -71.3376846, 65.8877106

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -47.6672821, 57.7376595, -91.4383774, 87.8462219
1: -25.9439163, 31.8255253, -36.8651505, 45.7337990, -71.6777115, 68.6906738
2: -22.5959740, 31.7507057, -32.1516266, 45.8342133, -68.4301910, 63.9023323
3: -31.1403046, 38.0657272, -44.6289749, 54.6052475, -85.7455521, 82.6947021
4: -29.2728157, 42.3466530, -41.7495308, 61.3076630, -90.5804749, 84.0961838

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305543, upper bound: 77.9399217
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9305543, upper bound: 77.9399217
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -47.6672821, 57.7376595, -93.0251312, 89.8834610
1: -27.1880989, 33.4509544, -36.8651505, 45.7337990, -72.9218903, 70.3160934
2: -23.6696396, 33.4155960, -32.1516266, 45.8342133, -69.5038528, 65.5672073
3: -32.6121635, 40.0253601, -44.6289749, 54.6052475, -87.2173996, 84.6543350
4: -30.6917725, 44.5793228, -41.7495308, 61.3076630, -91.9994354, 86.3288574

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9309351, upper bound: 77.9525363
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9309351, upper bound: 77.9528343
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -49.4725609, 59.7433586, -93.4440765, 89.6514969
1: -25.9439163, 31.8255253, -38.2433777, 47.3670807, -73.3109970, 70.0688934
2: -22.5959740, 31.7507057, -33.3768120, 47.4269867, -70.0229568, 65.1275177
3: -31.1403046, 38.0657272, -46.3069305, 56.5333519, -87.6736603, 84.3726349
4: -29.2728157, 42.3466530, -43.3185921, 63.4404526, -92.7132721, 85.6652451

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9394819, upper bound: 77.9381834
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382921, upper bound: 77.9382062
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -49.4725609, 59.7433586, -95.0308380, 91.6887512
1: -27.1880989, 33.4509544, -38.2433777, 47.3670807, -74.5551758, 71.6943207
2: -23.6696396, 33.4155960, -33.3768120, 47.4269867, -71.0966263, 66.7924042
3: -32.6121635, 40.0253601, -46.3069305, 56.5333519, -89.1455078, 86.3322906
4: -30.6917725, 44.5793228, -43.3185921, 63.4404526, -94.1322250, 87.8979187

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9398982, upper bound: 77.9517365
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387084, upper bound: 77.9517593
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -47.6672821, 57.7376595, -83.8493729, 78.1271210
1: -19.9750519, 24.0622005, -36.8651505, 45.7337990, -65.7088470, 60.9273529
2: -17.3923473, 24.0895405, -32.1516266, 45.8342133, -63.2265587, 56.2411652
3: -23.9274788, 28.7653179, -44.6289749, 54.6052475, -78.5327301, 73.3942947
4: -22.4313278, 32.1443291, -41.7495308, 61.3076630, -83.7389679, 73.8938599

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9285291, upper bound: 77.9402612
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9388661, upper bound: 77.9405842
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -38.3192291, 45.7232590, -47.6672821, 57.7376595, -96.0568848, 93.3905411
1: -29.4972591, 36.2548447, -36.8651505, 45.7337990, -75.2310410, 73.1199951
2: -25.7061634, 36.1928635, -32.1516266, 45.8342133, -71.5403748, 68.3444901
3: -35.4296150, 43.3433685, -44.6289749, 54.6052475, -90.0348587, 87.9723434
4: -33.3117332, 48.2953682, -41.7495308, 61.3076630, -94.6193771, 90.0448914

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9282720, upper bound: 77.9346543
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9386089, upper bound: 77.9349773
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -49.4725609, 59.7433586, -85.8550720, 79.9324036
1: -19.9750519, 24.0622005, -38.2433777, 47.3670807, -67.3421326, 62.3055763
2: -17.3923473, 24.0895405, -33.3768120, 47.4269867, -64.8193283, 57.4663544
3: -23.9274788, 28.7653179, -46.3069305, 56.5333519, -80.4608307, 75.0722351
4: -22.4313278, 32.1443291, -43.3185921, 63.4404526, -85.8717651, 75.4629211

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358407
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -38.3192291, 45.7232590, -49.4725609, 59.7433586, -98.0625916, 95.1958160
1: -29.4972591, 36.2548447, -38.2433777, 47.3670807, -76.8643417, 74.4982147
2: -25.7061634, 36.1928635, -33.3768120, 47.4269867, -73.1331329, 69.5696716
3: -35.4296150, 43.3433685, -46.3069305, 56.5333519, -91.9629669, 89.6502991
4: -33.3117332, 48.2953682, -43.3185921, 63.4404526, -96.7521820, 91.6139603

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358407
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -35.6116638, 42.6447182, -88.3093643, 90.7991791
1: -35.2989769, 43.7064095, -27.4477100, 33.7920532, -69.0910339, 71.1541214
2: -30.7957363, 43.7673340, -23.8948441, 33.7621422, -64.5578766, 67.6621780
3: -42.7660065, 52.1671257, -32.9314537, 40.4353333, -83.2013397, 85.0985489
4: -39.9625130, 58.5314178, -30.9905396, 45.0463181, -85.0088272, 89.5219574

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399217, upper bound: 77.9305543
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9399217, upper bound: 77.9305543
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -35.6116638, 42.6447182, -89.9679718, 92.9049377
1: -36.5915527, 45.3795662, -27.4477100, 33.7920532, -70.3836060, 72.8272781
2: -31.9135361, 45.4761581, -23.8948441, 33.7621422, -65.6756744, 69.3710022
3: -44.2947121, 54.1796265, -32.9314537, 40.4353333, -84.7300415, 87.1110535
4: -41.4348984, 60.8256683, -30.9905396, 45.0463181, -86.4811935, 91.8162003

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9402197, upper bound: 77.9396792
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9402197, upper bound: 77.9400600
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -38.8135300, 46.2989769, -91.9636230, 94.0010223
1: -35.2989769, 43.7064095, -29.8790951, 36.7157326, -72.0146942, 73.5855026
2: -30.7957363, 43.7673340, -26.0351105, 36.6458130, -67.4415436, 69.8024368
3: -42.7660065, 52.1671257, -35.8771858, 43.9032440, -86.6692505, 88.0443115
4: -39.9625130, 58.5314178, -33.7433815, 48.9063873, -88.8688965, 92.2747955

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9402612, upper bound: 77.9285291
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9346543, upper bound: 77.9282720
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -38.8135300, 46.2989769, -93.6222305, 96.1067963
1: -36.5915527, 45.3795662, -29.8790951, 36.7157326, -73.3072662, 75.2586594
2: -31.9135361, 45.4761581, -26.0351105, 36.6458130, -68.5593414, 71.5112686
3: -44.2947121, 54.1796265, -35.8771858, 43.9032440, -88.1979523, 90.0568085
4: -41.4348984, 60.8256683, -33.7433815, 48.9063873, -90.3412628, 94.5690460

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9405842, upper bound: 77.9388661
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9349773, upper bound: 77.9386089
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -35.6116638, 42.6447182, -78.3028717, 77.7928162
1: -27.3882751, 33.3255768, -27.4477100, 33.7920532, -61.1803284, 60.7732849
2: -23.9040565, 33.4101715, -23.8948441, 33.7621422, -57.6661987, 57.3050156
3: -33.1843910, 39.7667923, -32.9314537, 40.4353333, -73.6197205, 72.6982346
4: -30.8841171, 44.7131424, -30.9905396, 45.0463181, -75.9304123, 75.7036819

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381834, upper bound: 77.9394819
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381834, upper bound: 77.9398982
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -35.6116638, 42.6447182, -91.5671005, 94.7179489
1: -37.8203354, 46.8542061, -27.4477100, 33.7920532, -71.6123886, 74.3019180
2: -33.0129776, 46.9222374, -23.8948441, 33.7621422, -66.7751160, 70.8170776
3: -45.8136063, 55.9145012, -32.9314537, 40.4353333, -86.2489395, 88.8459320
4: -42.8412704, 62.7641792, -30.9905396, 45.0463181, -87.8875580, 93.7547150

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382062, upper bound: 77.9382921
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9517593, upper bound: 77.9387084
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -38.8135300, 46.2989769, -81.9571304, 80.9946594
1: -27.3882751, 33.3255768, -29.8790951, 36.7157326, -64.1040039, 63.2046700
2: -23.9040565, 33.4101715, -26.0351105, 36.6458130, -60.5498657, 59.4452820
3: -33.1843910, 39.7667923, -35.8771858, 43.9032440, -77.0876236, 75.6439819
4: -30.8841171, 44.7131424, -33.7433815, 48.9063873, -79.7904968, 78.4565277

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358407, upper bound: 77.9382668
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358407, upper bound: 77.9382668
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -38.8135300, 46.2989769, -95.2213593, 97.9197922
1: -37.8203354, 46.8542061, -29.8790951, 36.7157326, -74.5360565, 76.7332993
2: -33.0129776, 46.9222374, -26.0351105, 36.6458130, -69.6587830, 72.9573517
3: -45.8136063, 55.9145012, -35.8771858, 43.9032440, -89.7168503, 91.7916870
4: -42.8412704, 62.7641792, -33.7433815, 48.9063873, -91.7476273, 96.5075607

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9382668
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358765, upper bound: 77.9382668
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -47.6672821, 57.7376595, -103.4023132, 102.8547974
1: -35.2989769, 43.7064095, -36.8651505, 45.7337990, -81.0327454, 80.5715637
2: -30.7957363, 43.7673340, -32.1516266, 45.8342133, -76.6299515, 75.9189453
3: -42.7660065, 52.1671257, -44.6289749, 54.6052475, -97.3712540, 96.7960968
4: -39.9625130, 58.5314178, -41.7495308, 61.3076630, -101.2701721, 100.2809448

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9304314, upper bound: 77.9304314
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9304314, upper bound: 77.9307294
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -47.6672821, 57.7376595, -105.0609131, 104.9605560
1: -36.5915527, 45.3795662, -36.8651505, 45.7337990, -82.3253326, 82.2447128
2: -31.9135361, 45.4761581, -32.1516266, 45.8342133, -77.7477493, 77.6277771
3: -44.2947121, 54.1796265, -44.6289749, 54.6052475, -98.8999634, 98.8086014
4: -41.4348984, 60.8256683, -41.7495308, 61.3076630, -102.7425385, 102.5751877

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9307294, upper bound: 77.9395563
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9307294, upper bound: 77.9398544
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -49.4725609, 59.7433586, -105.4080124, 104.6600800
1: -35.2989769, 43.7064095, -38.2433777, 47.3670807, -82.6660461, 81.9497833
2: -30.7957363, 43.7673340, -33.3768120, 47.4269867, -78.2227173, 77.1441345
3: -42.7660065, 52.1671257, -46.3069305, 56.5333519, -99.2993622, 98.4740448
4: -39.9625130, 58.5314178, -43.3185921, 63.4404526, -103.4029694, 101.8500061

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393678, upper bound: 77.9285287
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381647, upper bound: 77.9282720
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -49.4725609, 59.7433586, -107.0666199, 106.7658310
1: -36.5915527, 45.3795662, -38.2433777, 47.3670807, -83.9586334, 83.6229324
2: -31.9135361, 45.4761581, -33.3768120, 47.4269867, -79.3405075, 78.8529663
3: -44.2947121, 54.1796265, -46.3069305, 56.5333519, -100.8280640, 100.4865494
4: -41.4348984, 60.8256683, -43.3185921, 63.4404526, -104.8753433, 104.1442566

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9396907, upper bound: 77.9388656
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384722, upper bound: 77.9386089
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -47.6672821, 57.7376595, -93.3958130, 89.8484344
1: -27.3882751, 33.3255768, -36.8651505, 45.7337990, -73.1220703, 70.1907272
2: -23.9040565, 33.4101715, -32.1516266, 45.8342133, -69.7382660, 65.5617905
3: -33.1843910, 39.7667923, -44.6289749, 54.6052475, -87.7896271, 84.3957672
4: -30.8841171, 44.7131424, -41.7495308, 61.3076630, -92.1917725, 86.4626694

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9285287, upper bound: 77.9393678
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9285287, upper bound: 77.9393678
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -47.6672821, 57.7376595, -106.6600418, 106.7735672
1: -37.8203354, 46.8542061, -36.8651505, 45.7337990, -83.5541306, 83.7193527
2: -33.0129776, 46.9222374, -32.1516266, 45.8342133, -78.8471909, 79.0738678
3: -45.8136063, 55.9145012, -44.6289749, 54.6052475, -100.4188538, 100.5434723
4: -42.8412704, 62.7641792, -41.7495308, 61.3076630, -104.1489105, 104.5137100

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9285515, upper bound: 77.9381779
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9388884, upper bound: 77.9385009
time: 1.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -49.4725609, 59.7433586, -95.4015198, 91.6537170
1: -27.3882751, 33.3255768, -38.2433777, 47.3670807, -74.7553558, 71.5689468
2: -23.9040565, 33.4101715, -33.3768120, 47.4269867, -71.3310318, 66.7869797
3: -33.1843910, 39.7667923, -46.3069305, 56.5333519, -89.7177429, 86.0737228
4: -30.8841171, 44.7131424, -43.3185921, 63.4404526, -94.3245697, 88.0317383

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9384835
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9384835
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -49.4725609, 59.7433586, -108.6657562, 108.5788422
1: -37.8203354, 46.8542061, -38.2433777, 47.3670807, -85.1874161, 85.0975723
2: -33.0129776, 46.9222374, -33.3768120, 47.4269867, -80.4399567, 80.2990494
3: -45.8136063, 55.9145012, -46.3069305, 56.5333519, -102.3469543, 102.2214279
4: -42.8412704, 62.7641792, -43.3185921, 63.4404526, -106.2817078, 106.0827713

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.35 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9400446, upper bound: 77.9400446
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9400446, upper bound: 77.9400446
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9404254, upper bound: 77.9526592
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9404254, upper bound: 77.9530400
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9403754, upper bound: 77.9381839
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9347685, upper bound: 77.9379267
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9407917, upper bound: 77.9517370
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9351848, upper bound: 77.9514798
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9381839, upper bound: 77.9403754
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9517370, upper bound: 77.9407917
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9356240
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9305543, upper bound: 77.9399217
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9305543, upper bound: 77.9399217
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9309351, upper bound: 77.9525363
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9309351, upper bound: 77.9528343
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9394819, upper bound: 77.9381834
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9382921, upper bound: 77.9382062
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9398982, upper bound: 77.9517365
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9387084, upper bound: 77.9517593
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9285291, upper bound: 77.9402612
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9388661, upper bound: 77.9405842
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9282720, upper bound: 77.9346543
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9386089, upper bound: 77.9349773
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358407
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358407
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9382668, upper bound: 77.9358765
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9399217, upper bound: 77.9305543
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9399217, upper bound: 77.9305543
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9402197, upper bound: 77.9396792
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9402197, upper bound: 77.9400600
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9402612, upper bound: 77.9285291
NS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9346543, upper bound: 77.9282720
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9405842, upper bound: 77.9388661
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9349773, upper bound: 77.9386089
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9381834, upper bound: 77.9394819
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9381834, upper bound: 77.9398982
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9382062, upper bound: 77.9382921
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9517593, upper bound: 77.9387084
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9358407, upper bound: 77.9382668
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9358407, upper bound: 77.9382668
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9356240, upper bound: 77.9382668
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9358765, upper bound: 77.9382668
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9304314, upper bound: 77.9304314
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9304314, upper bound: 77.9307294
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9307294, upper bound: 77.9395563
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9307294, upper bound: 77.9398544
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9393678, upper bound: 77.9285287
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9381647, upper bound: 77.9282720
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9396907, upper bound: 77.9388656
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9384722, upper bound: 77.9386089
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9285287, upper bound: 77.9393678
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9285287, upper bound: 77.9393678
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9285515, upper bound: 77.9381779
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9388884, upper bound: 77.9385009
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9384835
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9384835, upper bound: 77.9384835
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 3, lower bound: -77.9385194, upper bound: 77.9384835

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -33.7007179, 40.1789360, -73.8796539, 73.8796539
1: -25.9439163, 31.8255253, -25.9439163, 31.8255253, -57.7694397, 57.7694397
2: -22.5959740, 31.7507057, -22.5959740, 31.7507057, -54.3466759, 54.3466759
3: -31.1403046, 38.0657272, -31.1403046, 38.0657272, -69.2060165, 69.2060165
4: -29.2728157, 42.3466530, -29.2728157, 42.3466530, -71.6194687, 71.6194687

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9388106, upper bound: 77.9355044
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393834, upper bound: 77.9393834
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -35.2874756, 42.2161980, -75.9169006, 75.4664154
1: -25.9439163, 31.8255253, -27.1880989, 33.4509544, -59.3948708, 59.0136147
2: -22.5959740, 31.7507057, -23.6696396, 33.4155960, -56.0115662, 55.4203339
3: -31.1403046, 38.0657272, -32.6121635, 40.0253601, -71.1656647, 70.6778717
4: -29.2728157, 42.3466530, -30.6917725, 44.5793228, -73.8521423, 73.0384216

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9388106, upper bound: 77.9358852
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393834, upper bound: 77.9393834
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -33.7007179, 40.1789360, -75.4664154, 75.9169006
1: -27.1880989, 33.4509544, -25.9439163, 31.8255253, -59.0136147, 59.3948708
2: -23.6696396, 33.4155960, -22.5959740, 31.7507057, -55.4203377, 56.0115662
3: -32.6121635, 40.0253601, -31.1403046, 38.0657272, -70.6778717, 71.1656647
4: -30.6917725, 44.5793228, -29.2728157, 42.3466530, -73.0384216, 73.8521423

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395514, upper bound: 77.9514041
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9397642, upper bound: 77.9519447
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -35.2874756, 42.2161980, -77.5036774, 77.5036774
1: -27.1880989, 33.4509544, -27.1880989, 33.4509544, -60.6390457, 60.6390457
2: -23.6696396, 33.4155960, -23.6696396, 33.4155960, -57.0852242, 57.0852242
3: -32.6121635, 40.0253601, -32.6121635, 40.0253601, -72.6375122, 72.6375122
4: -30.6917725, 44.5793228, -30.6917725, 44.5793228, -75.2710953, 75.2710953

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395514, upper bound: 77.9517804
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395514, upper bound: 77.9523251
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -26.1117134, 30.4598427, -64.1605530, 66.2906494
1: -25.9439163, 31.8255253, -19.9750519, 24.0622005, -50.0061150, 51.8005753
2: -22.5959740, 31.7507057, -17.3923473, 24.0895405, -46.6855164, 49.1430511
3: -31.1403046, 38.0657272, -23.9274788, 28.7653179, -59.9056244, 61.9932022
4: -29.2728157, 42.3466530, -22.4313278, 32.1443291, -61.4171448, 64.7779694

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383419, upper bound: 77.9324075
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383419, upper bound: 77.9375546
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -26.1117134, 30.4598427, -65.7473145, 68.3279114
1: -27.1880989, 33.4509544, -19.9750519, 24.0622005, -51.2502899, 53.4260063
2: -23.6696396, 33.4155960, -17.3923473, 24.0895405, -47.7591743, 50.8079414
3: -32.6121635, 40.0253601, -23.9274788, 28.7653179, -61.3774796, 63.9528389
4: -30.6917725, 44.5793228, -22.4313278, 32.1443291, -62.8361015, 67.0106354

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9391316, upper bound: 77.9497429
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9395092, upper bound: 77.9510610
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -38.3192291, 45.7232590, -81.0107346, 80.5354309
1: -27.1880989, 33.4509544, -29.4972591, 36.2548447, -63.4429436, 62.9482117
2: -23.6696396, 33.4155960, -25.7061634, 36.1928635, -59.8624992, 59.1217575
3: -32.6121635, 40.0253601, -35.4296150, 43.3433685, -75.9555283, 75.4549713
4: -30.6917725, 44.5793228, -33.3117332, 48.2953682, -78.9871368, 77.8910522

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9334745, upper bound: 77.9494858
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9338522, upper bound: 77.9508038
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -33.7007179, 40.1789360, -66.2906494, 64.1605453
1: -19.9750519, 24.0622005, -25.9439163, 31.8255253, -51.8005753, 50.0061150
2: -17.3923473, 24.0895405, -22.5959740, 31.7507057, -49.1430511, 46.6855164
3: -23.9274788, 28.7653179, -31.1403046, 38.0657272, -61.9932022, 59.9056168
4: -22.4313278, 32.1443291, -29.2728157, 42.3466530, -64.7779694, 61.4171448

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9376264, upper bound: 77.9262055
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9376264, upper bound: 77.9403754
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -35.2874756, 42.2161980, -68.3279037, 65.7473068
1: -19.9750519, 24.0622005, -27.1880989, 33.4509544, -53.4260063, 51.2502899
2: -17.3923473, 24.0895405, -23.6696396, 33.4155960, -50.8079414, 47.7591743
3: -23.9274788, 28.7653179, -32.6121635, 40.0253601, -63.9528389, 61.3774757
4: -22.4313278, 32.1443291, -30.6917725, 44.5793228, -67.0106354, 62.8360939

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9511795, upper bound: 77.9266218
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9511795, upper bound: 77.9407917
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -45.6646538, 55.1875191, -88.8882370, 85.8435898
1: -25.9439163, 31.8255253, -35.2989769, 43.7064095, -69.6503296, 67.1244736
2: -22.5959740, 31.7507057, -30.7957363, 43.7673340, -66.3633118, 62.5464401
3: -31.1403046, 38.0657272, -42.7660065, 52.1671257, -83.3074265, 80.8317261
4: -29.2728157, 42.3466530, -39.9625130, 58.5314178, -87.8042297, 82.3091660

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9289732, upper bound: 77.9353815
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9295460, upper bound: 77.9392604
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -47.3232613, 57.2932701, -90.9939880, 87.5021973
1: -25.9439163, 31.8255253, -36.5915527, 45.3795662, -71.3234787, 68.4170609
2: -22.5959740, 31.7507057, -31.9135361, 45.4761581, -68.0721283, 63.6642418
3: -31.1403046, 38.0657272, -44.2947121, 54.1796265, -85.3199158, 82.3604202
4: -29.2728157, 42.3466530, -41.4348984, 60.8256683, -90.0984802, 83.7815399

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9289732, upper bound: 77.9353815
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9295460, upper bound: 77.9392604
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -45.6646538, 55.1875191, -90.4749908, 87.8808365
1: -27.1880989, 33.4509544, -35.2989769, 43.7064095, -70.8945084, 68.7499084
2: -23.6696396, 33.4155960, -30.7957363, 43.7673340, -67.4369736, 64.2113342
3: -32.6121635, 40.0253601, -42.7660065, 52.1671257, -84.7792664, 82.7913666
4: -30.6917725, 44.5793228, -39.9625130, 58.5314178, -89.2231903, 84.5418396

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9297141, upper bound: 77.9512812
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9299268, upper bound: 77.9518218
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -47.3232613, 57.2932701, -92.5807495, 89.5394363
1: -27.1880989, 33.4509544, -36.5915527, 45.3795662, -72.5676575, 70.0424805
2: -23.6696396, 33.4155960, -31.9135361, 45.4761581, -69.1457977, 65.3291321
3: -32.6121635, 40.0253601, -44.2947121, 54.1796265, -86.7917709, 84.3200684
4: -30.6917725, 44.5793228, -41.4348984, 60.8256683, -91.5174408, 86.0142136

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9297141, upper bound: 77.9512812
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9299268, upper bound: 77.9521199
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -35.6581573, 42.1811523, -75.8818665, 75.8370972
1: -25.9439163, 31.8255253, -27.3882751, 33.3255768, -59.2694893, 59.2137909
2: -22.5959740, 31.7507057, -23.9040565, 33.4101715, -56.0061455, 55.6547623
3: -31.1403046, 38.0657272, -33.1843910, 39.7667923, -70.9070969, 71.2500916
4: -29.2728157, 42.3466530, -30.8841171, 44.7131424, -73.9859619, 73.2307587

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9374317, upper bound: 77.9324070
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381827, upper bound: 77.9375541
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -33.7007179, 40.1789360, -48.9223938, 59.1062851, -92.8069992, 89.1013336
1: -25.9439163, 31.8255253, -37.8203354, 46.8542061, -72.7981186, 69.6458435
2: -22.5959740, 31.7507057, -33.0129776, 46.9222374, -69.5182114, 64.7636871
3: -31.1403046, 38.0657272, -45.8136063, 55.9145012, -87.0548096, 83.8793259
4: -29.2728157, 42.3466530, -42.8412704, 62.7641792, -92.0369949, 85.1879044

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9363760, upper bound: 77.9324298
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9371270, upper bound: 77.9375769
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -35.6581573, 42.1811523, -77.4686279, 77.8743439
1: -27.1880989, 33.4509544, -27.3882751, 33.3255768, -60.5136757, 60.8392258
2: -23.6696396, 33.4155960, -23.9040565, 33.4101715, -57.0798035, 57.3196487
3: -32.6121635, 40.0253601, -33.1843910, 39.7667923, -72.3789520, 73.2097473
4: -30.6917725, 44.5793228, -30.8841171, 44.7131424, -75.4049149, 75.4634399

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9382214, upper bound: 77.9497424
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385990, upper bound: 77.9510605
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -35.2874756, 42.2161980, -48.9223938, 59.1062851, -94.3937607, 91.1385651
1: -27.1880989, 33.4509544, -37.8203354, 46.8542061, -74.0422974, 71.2712708
2: -23.6696396, 33.4155960, -33.0129776, 46.9222374, -70.5918732, 66.4285736
3: -32.6121635, 40.0253601, -45.8136063, 55.9145012, -88.5266495, 85.8389664
4: -30.6917725, 44.5793228, -42.8412704, 62.7641792, -93.4559479, 87.4205780

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9371657, upper bound: 77.9497652
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9375433, upper bound: 77.9510833
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -45.6646538, 55.1875191, -81.2992325, 76.1244812
1: -19.9750519, 24.0622005, -35.2989769, 43.7064095, -63.6814613, 59.3611755
2: -17.3923473, 24.0895405, -30.7957363, 43.7673340, -61.1596680, 54.8852768
3: -23.9274788, 28.7653179, -42.7660065, 52.1671257, -76.0946045, 71.5313263
4: -22.4313278, 32.1443291, -39.9625130, 58.5314178, -80.9627304, 72.1068420

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9279716, upper bound: 77.9260913
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9279716, upper bound: 77.9402612
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -47.3232613, 57.2932701, -83.4049835, 77.7830963
1: -19.9750519, 24.0622005, -36.5915527, 45.3795662, -65.3546143, 60.6537552
2: -17.3923473, 24.0895405, -31.9135361, 45.4761581, -62.8685036, 56.0030746
3: -23.9274788, 28.7653179, -44.2947121, 54.1796265, -78.1071014, 73.0600281
4: -22.4313278, 32.1443291, -41.4348984, 60.8256683, -83.2569580, 73.5792236

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383086, upper bound: 77.9264143
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383086, upper bound: 77.9405842
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -38.3192291, 45.7232590, -47.3232613, 57.2932701, -95.6125031, 93.0465240
1: -29.4972591, 36.2548447, -36.5915527, 45.3795662, -74.8768158, 72.8463898
2: -25.7061634, 36.1928635, -31.9135361, 45.4761581, -71.1823120, 68.1063995
3: -35.4296150, 43.3433685, -44.2947121, 54.1796265, -89.6092224, 87.6380768
4: -33.3117332, 48.2953682, -41.4348984, 60.8256683, -94.1373901, 89.7302475

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383421, upper bound: 77.9254394
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383421, upper bound: 77.9349773
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -35.6581573, 42.1811523, -68.2928619, 66.1179810
1: -19.9750519, 24.0622005, -27.3882751, 33.3255768, -53.3006287, 51.4504700
2: -17.3923473, 24.0895405, -23.9040565, 33.4101715, -50.8025208, 47.9935989
3: -23.9274788, 28.7653179, -33.1843910, 39.7667923, -63.6942711, 61.9497070
4: -22.4313278, 32.1443291, -30.8841171, 44.7131424, -67.1444473, 63.0284462

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9377471, upper bound: 77.9262048
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383046, upper bound: 77.9403747
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -26.1117134, 30.4598427, -48.9223938, 59.1062851, -85.2180023, 79.3822174
1: -19.9750519, 24.0622005, -37.8203354, 46.8542061, -66.8292542, 61.8825340
2: -17.3923473, 24.0895405, -33.0129776, 46.9222374, -64.3145828, 57.1025162
3: -23.9274788, 28.7653179, -45.8136063, 55.9145012, -79.8419800, 74.5789261
4: -22.4313278, 32.1443291, -42.8412704, 62.7641792, -85.1955032, 74.9855881

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9377471, upper bound: 77.9262404
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383046, upper bound: 77.9404102
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -38.3192291, 45.7232590, -35.6581573, 42.1811523, -80.5003815, 81.3814163
1: -29.4972591, 36.2548447, -27.3882751, 33.3255768, -62.8228378, 63.6431198
2: -25.7061634, 36.1928635, -23.9040565, 33.4101715, -59.1163330, 60.0969200
3: -35.4296150, 43.3433685, -33.1843910, 39.7667923, -75.1964035, 76.5277557
4: -33.3117332, 48.2953682, -30.8841171, 44.7131424, -78.0248718, 79.1794739

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9377806, upper bound: 77.9252300
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9380475, upper bound: 77.9347678
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -38.3192291, 45.7232590, -48.9223938, 59.1062851, -97.4255142, 94.6456528
1: -29.4972591, 36.2548447, -37.8203354, 46.8542061, -76.3514481, 74.0751724
2: -25.7061634, 36.1928635, -33.0129776, 46.9222374, -72.6284027, 69.2058411
3: -35.4296150, 43.3433685, -45.8136063, 55.9145012, -91.3441086, 89.1569748
4: -33.3117332, 48.2953682, -42.8412704, 62.7641792, -96.0759125, 91.1366119

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9377806, upper bound: 77.9252440
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9380475, upper bound: 77.9347678
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -33.7007179, 40.1789360, -85.8435898, 88.8882370
1: -35.2989769, 43.7064095, -25.9439163, 31.8255253, -67.1244812, 69.6503296
2: -30.7957363, 43.7673340, -22.5959740, 31.7507057, -62.5464401, 66.3633118
3: -42.7660065, 52.1671257, -31.1403046, 38.0657272, -80.8317261, 83.3074265
4: -39.9625130, 58.5314178, -29.2728157, 42.3466530, -82.3091660, 87.8042297

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385975, upper bound: 77.9239898
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385975, upper bound: 77.9295460
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -35.2874756, 42.2161980, -87.8808365, 90.4749908
1: -35.2989769, 43.7064095, -27.1880989, 33.4509544, -68.7499008, 70.8945084
2: -30.7957363, 43.7673340, -23.6696396, 33.4155960, -64.2113342, 67.4369736
3: -42.7660065, 52.1671257, -32.6121635, 40.0253601, -82.7913666, 84.7792664
4: -39.9625130, 58.5314178, -30.6917725, 44.5793228, -84.5418396, 89.2231903

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9385975, upper bound: 77.9239898
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9392605, upper bound: 77.9295460
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -33.7007179, 40.1789360, -87.5021973, 90.9939880
1: -36.5915527, 45.3795662, -25.9439163, 31.8255253, -68.4170609, 71.3234787
2: -31.9135361, 45.4761581, -22.5959740, 31.7507057, -63.6642418, 68.0721283
3: -44.2947121, 54.1796265, -31.1403046, 38.0657272, -82.3604202, 85.3199310
4: -41.4348984, 60.8256683, -29.2728157, 42.3466530, -83.7815399, 90.0984802

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9370036, upper bound: 77.8787656
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393781, upper bound: 77.9396442
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -35.2874756, 42.2161980, -89.5394363, 92.5807495
1: -36.5915527, 45.3795662, -27.1880989, 33.4509544, -70.0424805, 72.5676575
2: -31.9135361, 45.4761581, -23.6696396, 33.4155960, -65.3291321, 69.1457977
3: -44.2947121, 54.1796265, -32.6121635, 40.0253601, -84.3200684, 86.7917633
4: -41.4348984, 60.8256683, -30.6917725, 44.5793228, -86.0142136, 91.5174408

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9370155, upper bound: 77.8787656
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393781, upper bound: 77.9396442
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -26.1117134, 30.4598427, -76.1244812, 81.2992325
1: -35.2989769, 43.7064095, -19.9750519, 24.0622005, -59.3611755, 63.6814613
2: -30.7957363, 43.7673340, -17.3923473, 24.0895405, -54.8852768, 61.1596832
3: -42.7660065, 52.1671257, -23.9274788, 28.7653179, -71.5313187, 76.0946045
4: -39.9625130, 58.5314178, -22.4313278, 32.1443291, -72.1068420, 80.9627304

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9381298, upper bound: 77.9205247
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9389787, upper bound: 77.9277160
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -26.1117134, 30.4598427, -77.7830963, 83.4049835
1: -36.5915527, 45.3795662, -19.9750519, 24.0622005, -60.6537552, 65.3546143
2: -31.9135361, 45.4761581, -17.3923473, 24.0895405, -56.0030746, 62.8685074
3: -44.2947121, 54.1796265, -23.9274788, 28.7653179, -73.0600281, 78.1071014
4: -41.4348984, 60.8256683, -22.4313278, 32.1443291, -73.5792236, 83.2569656

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9387245, upper bound: 77.9356786
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9393017, upper bound: 77.9377655
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -38.3192291, 45.7232590, -93.0465240, 95.6125031
1: -36.5915527, 45.3795662, -29.4972591, 36.2548447, -72.8463974, 74.8768158
2: -31.9135361, 45.4761581, -25.7061634, 36.1928635, -68.1063995, 71.1823120
3: -44.2947121, 54.1796265, -35.4296150, 43.3433685, -87.6380768, 89.6092224
4: -41.4348984, 60.8256683, -33.3117332, 48.2953682, -89.7302475, 94.1373901

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9330675, upper bound: 77.9354215
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9330675, upper bound: 77.9375084
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -33.7007179, 40.1789360, -75.8370972, 75.8818665
1: -27.3882751, 33.3255768, -25.9439163, 31.8255253, -59.2137947, 59.2694893
2: -23.9040565, 33.4101715, -22.5959740, 31.7507057, -55.6547623, 56.0061455
3: -33.1843910, 39.7667923, -31.1403046, 38.0657272, -71.2501144, 70.9070969
4: -30.8841171, 44.7131424, -29.2728157, 42.3466530, -73.2307587, 73.9859619

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9373297, upper bound: 77.9368536
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9375541, upper bound: 77.9381827
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -35.2874756, 42.2161980, -77.8743362, 77.4686279
1: -27.3882751, 33.3255768, -27.1880989, 33.4509544, -60.8392181, 60.5136604
2: -23.9040565, 33.4101715, -23.6696396, 33.4155960, -57.3196487, 57.0798035
3: -33.1843910, 39.7667923, -32.6121635, 40.0253601, -73.2097473, 72.3789520
4: -30.8841171, 44.7131424, -30.6917725, 44.5793228, -75.4634323, 75.4049149

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9508360, upper bound: 77.9372699
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9510605, upper bound: 77.9385990
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -33.7007179, 40.1789360, -89.1013336, 92.8069992
1: -37.8203354, 46.8542061, -25.9439163, 31.8255253, -69.6458435, 72.7981186
2: -33.0129776, 46.9222374, -22.5959740, 31.7507057, -64.7636871, 69.5182114
3: -45.8136063, 55.9145012, -31.1403046, 38.0657272, -83.8793259, 87.0548096
4: -42.8412704, 62.7641792, -29.2728157, 42.3466530, -85.1879044, 92.0369949

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378204, upper bound: 77.9262072
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9378204, upper bound: 77.9382921
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -35.2874756, 42.2161980, -91.1385651, 94.3937607
1: -37.8203354, 46.8542061, -27.1880989, 33.4509544, -71.2712708, 74.0422974
2: -33.0129776, 46.9222374, -23.6696396, 33.4155960, -66.4285660, 70.5918732
3: -45.8136063, 55.9145012, -32.6121635, 40.0253601, -85.8389664, 88.5266495
4: -42.8412704, 62.7641792, -30.6917725, 44.5793228, -87.4205780, 93.4559479

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9513735, upper bound: 77.9266235
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9513735, upper bound: 77.9387084
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -26.1117134, 30.4598427, -66.1179810, 68.2928619
1: -27.3882751, 33.3255768, -19.9750519, 24.0622005, -51.4504700, 53.3006287
2: -23.9040565, 33.4101715, -17.3923473, 24.0895405, -47.9935989, 50.8025169
3: -33.1843910, 39.7667923, -23.9274788, 28.7653179, -61.9497070, 63.6942711
4: -30.8841171, 44.7131424, -22.4313278, 32.1443291, -63.0284462, 67.1444550

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9343165, upper bound: 77.9368690
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9345407, upper bound: 77.9381701
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -38.3192291, 45.7232590, -81.3814163, 80.5003815
1: -27.3882751, 33.3255768, -29.4972591, 36.2548447, -63.6431198, 62.8228378
2: -23.9040565, 33.4101715, -25.7061634, 36.1928635, -60.0969200, 59.1163330
3: -33.1843910, 39.7667923, -35.4296150, 43.3433685, -76.5277557, 75.1964035
4: -30.8841171, 44.7131424, -33.3117332, 48.2953682, -79.1794739, 78.0248718

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9343165, upper bound: 77.9368690
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9343165, upper bound: 77.9381701
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -26.1117134, 30.4598427, -79.3822174, 85.2180023
1: -37.8203354, 46.8542061, -19.9750519, 24.0622005, -61.8825302, 66.8292542
2: -33.0129776, 46.9222374, -17.3923473, 24.0895405, -57.1025162, 64.3145828
3: -45.8136063, 55.9145012, -23.9274788, 28.7653179, -74.5789261, 79.8419800
4: -42.8412704, 62.7641792, -22.4313278, 32.1443291, -74.9855881, 85.1955032

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9344175, upper bound: 77.9259626
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9347678, upper bound: 77.9380474
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -38.3192291, 45.7232590, -94.6456528, 97.4255142
1: -37.8203354, 46.8542061, -29.4972591, 36.2548447, -74.0751724, 76.3514481
2: -33.0129776, 46.9222374, -25.7061634, 36.1928635, -69.2058411, 72.6284027
3: -45.8136063, 55.9145012, -35.4296150, 43.3433685, -89.1569748, 91.3441086
4: -42.8412704, 62.7641792, -33.3117332, 48.2953682, -91.1366119, 96.0759125

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9344175, upper bound: 77.9259626
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9348034, upper bound: 77.9380474
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -45.6646538, 55.1875191, -102.5107803, 102.9579239
1: -36.5915527, 45.3795662, -35.2989769, 43.7064095, -80.2979584, 80.6785202
2: -31.9135361, 45.4761581, -30.7957363, 43.7673340, -75.6808548, 76.2718964
3: -44.2947121, 54.1796265, -42.7660065, 52.1671257, -96.4618301, 96.9456329
4: -41.4348984, 60.8256683, -39.9625130, 58.5314178, -99.9663086, 100.7881775

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275286, upper bound: 77.8786466
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9299032, upper bound: 77.9395252
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -47.3232613, 57.2932701, -104.6165314, 104.6165314
1: -36.5915527, 45.3795662, -36.5915527, 45.3795662, -81.9711151, 81.9711151
2: -31.9135361, 45.4761581, -31.9135361, 45.4761581, -77.3896866, 77.3896866
3: -44.2947121, 54.1796265, -44.2947121, 54.1796265, -98.4743271, 98.4743347
4: -41.4348984, 60.8256683, -41.4348984, 60.8256683, -102.2605438, 102.2605362

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9275405, upper bound: 77.8786466
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9299032, upper bound: 77.9398313
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -35.6581573, 42.1811523, -87.8458099, 90.8456726
1: -35.2989769, 43.7064095, -27.3882751, 33.3255768, -68.6245270, 71.0946808
2: -30.7957363, 43.7673340, -23.9040565, 33.4101715, -64.2059097, 67.6713867
3: -42.7660065, 52.1671257, -33.1843910, 39.7667923, -82.5327988, 85.3515015
4: -39.9625130, 58.5314178, -30.8841171, 44.7131424, -84.6756592, 89.4155273

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9372196, upper bound: 77.9205243
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9380685, upper bound: 77.9277155
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -45.6646538, 55.1875191, -48.9223938, 59.1062851, -104.7709351, 104.1099091
1: -35.2989769, 43.7064095, -37.8203354, 46.8542061, -82.1531601, 81.5267487
2: -30.7957363, 43.7673340, -33.0129776, 46.9222374, -77.7179718, 76.7803040
3: -42.7660065, 52.1671257, -45.8136063, 55.9145012, -98.6805115, 97.9807281
4: -39.9625130, 58.5314178, -42.8412704, 62.7641792, -102.7266922, 101.3726730

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9361430, upper bound: 77.9202676
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9369997, upper bound: 77.9274588
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -35.6581573, 42.1811523, -89.5044098, 92.9514313
1: -36.5915527, 45.3795662, -27.3882751, 33.3255768, -69.9171143, 72.7678375
2: -31.9135361, 45.4761581, -23.9040565, 33.4101715, -65.3237076, 69.3802185
3: -44.2947121, 54.1796265, -33.1843910, 39.7667923, -84.0615082, 87.3639984
4: -41.4348984, 60.8256683, -30.8841171, 44.7131424, -86.1480255, 91.7097549

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9378143, upper bound: 77.9356782
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383915, upper bound: 77.9377651
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.3232613, 57.2932701, -48.9223938, 59.1062851, -106.4295425, 106.2156677
1: -36.5915527, 45.3795662, -37.8203354, 46.8542061, -83.4457550, 83.1998901
2: -31.9135361, 45.4761581, -33.0129776, 46.9222374, -78.8357697, 78.4891357
3: -44.2947121, 54.1796265, -45.8136063, 55.9145012, -100.2092133, 99.9932327
4: -41.4348984, 60.8256683, -42.8412704, 62.7641792, -104.1990814, 103.6669083

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9367299, upper bound: 77.9354215
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9369997, upper bound: 77.9375084
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -45.6646538, 55.1875191, -90.8456726, 87.8458099
1: -27.3882751, 33.3255768, -35.2989769, 43.7064095, -71.0946808, 68.6245422
2: -23.9040565, 33.4101715, -30.7957363, 43.7673340, -67.6713867, 64.2059097
3: -33.1843910, 39.7667923, -42.7660065, 52.1671257, -85.3515015, 82.5327988
4: -30.8841171, 44.7131424, -39.9625130, 58.5314178, -89.4155273, 84.6756592

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9274911, upper bound: 77.9367395
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9277155, upper bound: 77.9380685
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -47.3232613, 57.2932701, -92.9514313, 89.5044098
1: -27.3882751, 33.3255768, -36.5915527, 45.3795662, -72.7678375, 69.9171219
2: -23.9040565, 33.4101715, -31.9135361, 45.4761581, -69.3802185, 65.3237076
3: -33.1843910, 39.7667923, -44.2947121, 54.1796265, -87.3639908, 84.0615082
4: -30.8841171, 44.7131424, -41.4348984, 60.8256683, -91.7097626, 86.1480255

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9375406, upper bound: 77.9370624
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9377651, upper bound: 77.9383915
time: 2.16 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -45.6646538, 55.1875191, -104.1099091, 104.7709351
1: -37.8203354, 46.8542061, -35.2989769, 43.7064095, -81.5267487, 82.1531601
2: -33.0129776, 46.9222374, -30.7957363, 43.7673340, -76.7803040, 77.7179718
3: -45.8136063, 55.9145012, -42.7660065, 52.1671257, -97.9807281, 98.6805115
4: -42.8412704, 62.7641792, -39.9625130, 58.5314178, -101.3726730, 102.7266922

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9281656, upper bound: 77.9260930
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9281656, upper bound: 77.9381779
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -47.3232613, 57.2932701, -106.2156677, 106.4295425
1: -37.8203354, 46.8542061, -36.5915527, 45.3795662, -83.1998901, 83.4457474
2: -33.0129776, 46.9222374, -31.9135361, 45.4761581, -78.4891357, 78.8357697
3: -45.8136063, 55.9145012, -44.2947121, 54.1796265, -99.9932327, 100.2092133
4: -42.8412704, 62.7641792, -41.4348984, 60.8256683, -103.6669006, 104.1990814

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383013, upper bound: 77.9264160
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383013, upper bound: 77.9385009
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -35.6581573, 42.1811523, -77.8393097, 77.8393097
1: -27.3882751, 33.3255768, -27.3882751, 33.3255768, -60.7138481, 60.7138519
2: -23.9040565, 33.4101715, -23.9040565, 33.4101715, -57.3142281, 57.3142281
3: -33.1843910, 39.7667923, -33.1843910, 39.7667923, -72.9511871, 72.9511871
4: -30.8841171, 44.7131424, -30.8841171, 44.7131424, -75.5972443, 75.5972519

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9370955, upper bound: 77.9370327
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9373185, upper bound: 77.9381701
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -35.6581573, 42.1811523, -48.9223938, 59.1062851, -94.7644348, 91.1035385
1: -27.3882751, 33.3255768, -37.8203354, 46.8542061, -74.2424774, 71.1459045
2: -23.9040565, 33.4101715, -33.0129776, 46.9222374, -70.8262939, 66.4231491
3: -33.1843910, 39.7667923, -45.8136063, 55.9145012, -89.0988770, 85.5803986
4: -30.8841171, 44.7131424, -42.8412704, 62.7641792, -93.6483002, 87.5543900

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9370955, upper bound: 77.9370327
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9370955, upper bound: 77.9381701
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -35.6581573, 42.1811523, -91.1035385, 94.7644424
1: -37.8203354, 46.8542061, -27.3882751, 33.3255768, -71.1459045, 74.2424774
2: -33.0129776, 46.9222374, -23.9040565, 33.4101715, -66.4231491, 70.8262939
3: -45.8136063, 55.9145012, -33.1843910, 39.7667923, -85.5803986, 89.0988770
4: -42.8412704, 62.7641792, -30.8841171, 44.7131424, -87.5543900, 93.6483002

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9377237, upper bound: 77.9262065
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383269, upper bound: 77.9382914
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.9223938, 59.1062851, -48.9223938, 59.1062851, -108.0286713, 108.0286713
1: -37.8203354, 46.8542061, -37.8203354, 46.8542061, -84.6745300, 84.6745300
2: -33.0129776, 46.9222374, -33.0129776, 46.9222374, -79.9352112, 79.9352112
3: -45.8136063, 55.9145012, -45.8136063, 55.9145012, -101.7281036, 101.7281036
4: -42.8412704, 62.7641792, -42.8412704, 62.7641792, -105.6054459, 105.6054459

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9379411, upper bound: 77.9262421
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9383269, upper bound: 77.9382914
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.66 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9388106, upper bound: 77.9355044
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9393834, upper bound: 77.9393834
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9388106, upper bound: 77.9358852
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9393834, upper bound: 77.9393834
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9395514, upper bound: 77.9514041
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9397642, upper bound: 77.9519447
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9395514, upper bound: 77.9517804
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9395514, upper bound: 77.9523251
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383419, upper bound: 77.9324075
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383419, upper bound: 77.9375546
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9391316, upper bound: 77.9497429
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9395092, upper bound: 77.9510610
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9334745, upper bound: 77.9494858
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9338522, upper bound: 77.9508038
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9376264, upper bound: 77.9262055
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9376264, upper bound: 77.9403754
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9511795, upper bound: 77.9266218
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9511795, upper bound: 77.9407917
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9289732, upper bound: 77.9353815
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9295460, upper bound: 77.9392604
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9289732, upper bound: 77.9353815
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9295460, upper bound: 77.9392604
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9297141, upper bound: 77.9512812
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9299268, upper bound: 77.9518218
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9297141, upper bound: 77.9512812
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9299268, upper bound: 77.9521199
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9374317, upper bound: 77.9324070
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9381827, upper bound: 77.9375541
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9363760, upper bound: 77.9324298
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9371270, upper bound: 77.9375769
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9382214, upper bound: 77.9497424
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9385990, upper bound: 77.9510605
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9371657, upper bound: 77.9497652
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9375433, upper bound: 77.9510833
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9279716, upper bound: 77.9260913
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9279716, upper bound: 77.9402612
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383086, upper bound: 77.9264143
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383086, upper bound: 77.9405842
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383421, upper bound: 77.9254394
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383421, upper bound: 77.9349773
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9377471, upper bound: 77.9262048
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383046, upper bound: 77.9403747
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9377471, upper bound: 77.9262404
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383046, upper bound: 77.9404102
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9377806, upper bound: 77.9252300
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9380475, upper bound: 77.9347678
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9377806, upper bound: 77.9252440
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9380475, upper bound: 77.9347678
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9385975, upper bound: 77.9239898
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9385975, upper bound: 77.9295460
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9385975, upper bound: 77.9239898
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9392605, upper bound: 77.9295460
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9370036, upper bound: 77.8787656
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9393781, upper bound: 77.9396442
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9370155, upper bound: 77.8787656
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9393781, upper bound: 77.9396442
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9381298, upper bound: 77.9205247
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9389787, upper bound: 77.9277160
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9387245, upper bound: 77.9356786
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9393017, upper bound: 77.9377655
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9330675, upper bound: 77.9354215
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9330675, upper bound: 77.9375084
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9373297, upper bound: 77.9368536
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9375541, upper bound: 77.9381827
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9508360, upper bound: 77.9372699
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9510605, upper bound: 77.9385990
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9378204, upper bound: 77.9262072
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9378204, upper bound: 77.9382921
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9513735, upper bound: 77.9266235
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9513735, upper bound: 77.9387084
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9343165, upper bound: 77.9368690
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9345407, upper bound: 77.9381701
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9343165, upper bound: 77.9368690
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9343165, upper bound: 77.9381701
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9344175, upper bound: 77.9259626
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9347678, upper bound: 77.9380474
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9344175, upper bound: 77.9259626
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9348034, upper bound: 77.9380474
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9275286, upper bound: 77.8786466
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9299032, upper bound: 77.9395252
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9275405, upper bound: 77.8786466
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9299032, upper bound: 77.9398313
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9372196, upper bound: 77.9205243
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9380685, upper bound: 77.9277155
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9361430, upper bound: 77.9202676
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9369997, upper bound: 77.9274588
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9378143, upper bound: 77.9356782
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383915, upper bound: 77.9377651
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9367299, upper bound: 77.9354215
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9369997, upper bound: 77.9375084
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9274911, upper bound: 77.9367395
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9277155, upper bound: 77.9380685
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9375406, upper bound: 77.9370624
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9377651, upper bound: 77.9383915
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9281656, upper bound: 77.9260930
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9281656, upper bound: 77.9381779
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383013, upper bound: 77.9264160
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383013, upper bound: 77.9385009
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9370955, upper bound: 77.9370327
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9373185, upper bound: 77.9381701
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9370955, upper bound: 77.9370327
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9370955, upper bound: 77.9381701
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9377237, upper bound: 77.9262065
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383269, upper bound: 77.9382914
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9379411, upper bound: 77.9262421
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -77.9383269, upper bound: 77.9382914

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -31.1855011, 37.4674988, -33.7007179, 40.1789360, -71.3644333, 71.1682129
1: -24.1073112, 29.6677723, -25.9439163, 31.8255253, -55.9328384, 55.6116829
2: -20.9856720, 29.6444969, -22.5959740, 31.7507057, -52.7363739, 52.2404709
3: -28.8964233, 35.5232506, -31.1403046, 38.0657272, -66.9621277, 66.6635590
4: -27.2243156, 39.5415764, -29.2728157, 42.3466530, -69.5709610, 68.8143845

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9349316, upper bound: 77.9349316
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9349316, upper bound: 77.9355044
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -33.2685127, 39.6535950, -33.7007179, 40.1789360, -73.4474487, 73.3543091
1: -25.6111336, 31.4002991, -25.9439163, 31.8255253, -57.4366608, 57.3442154
2: -22.3033543, 31.3242931, -22.5959740, 31.7507057, -54.0540504, 53.9202652
3: -30.7320042, 37.5634918, -31.1403046, 38.0657272, -68.7977295, 68.7037964
4: -28.8902245, 41.7764015, -29.2728157, 42.3466530, -71.2368774, 71.0492172

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9355044, upper bound: 77.9388104
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9355044, upper bound: 77.9393834
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -31.1855011, 37.4674988, -35.2874756, 42.2161980, -73.4016800, 72.7549744
1: -24.1073112, 29.6677723, -27.1880989, 33.4509544, -57.5582657, 56.8558578
2: -20.9856720, 29.6444969, -23.6696396, 33.4155960, -54.4012642, 53.3141365
3: -28.8964233, 35.5232506, -32.6121635, 40.0253601, -68.9217758, 68.1354141
4: -27.2243156, 39.5415764, -30.6917725, 44.5793228, -71.8036270, 70.2333527

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9508313, upper bound: 77.9356725
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9508313, upper bound: 77.9358852
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -33.2685127, 39.6535950, -35.2874756, 42.2161980, -75.4846954, 74.9410706
1: -25.6111336, 31.4002991, -27.1880989, 33.4509544, -59.0620842, 58.5883904
2: -22.3033543, 31.3242931, -23.6696396, 33.4155960, -55.7189369, 54.9939232
3: -30.7320042, 37.5634918, -32.6121635, 40.0253601, -70.7573624, 70.1756592
4: -28.8902245, 41.7764015, -30.6917725, 44.5793228, -73.4695435, 72.4681702

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9514041, upper bound: 77.9395514
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9514041, upper bound: 77.9397642
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -32.8798866, 39.6466942, -33.7007179, 40.1789360, -73.0588226, 73.3473969
1: -25.4290619, 31.4012699, -25.9439163, 31.8255253, -57.2545853, 57.3451843
2: -22.1278496, 31.4176846, -22.5959740, 31.7507057, -53.8785553, 54.0136566
3: -30.4659748, 37.6071281, -31.1403046, 38.0657272, -68.5317001, 68.7474289
4: -28.7335110, 41.9202194, -29.2728157, 42.3466530, -71.0801620, 71.1930313

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9356725, upper bound: 77.9508313
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9356725, upper bound: 77.9514041
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -34.8573303, 41.6929207, -33.7007179, 40.1789360, -75.0362701, 75.3936310
1: -26.8566914, 33.0270042, -25.9439163, 31.8255253, -58.6822166, 58.9709206
2: -23.3781586, 32.9910622, -22.5959740, 31.7507057, -55.1288643, 55.5870323
3: -32.2049904, 39.5241165, -31.1403046, 38.0657272, -70.2706833, 70.6644211
4: -30.3103943, 44.0107918, -29.2728157, 42.3466530, -72.6570435, 73.2836075

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358852, upper bound: 77.9513717
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9358852, upper bound: 77.9519447
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -32.8798866, 39.6466942, -35.2874756, 42.2161980, -75.0960693, 74.9341736
1: -25.4290619, 31.4012699, -27.1880989, 33.4509544, -58.8800163, 58.5893593
2: -22.1278496, 31.4176846, -23.6696396, 33.4155960, -55.5434418, 55.0873222
3: -30.4659748, 37.6071281, -32.6121635, 40.0253601, -70.4913330, 70.2192841
4: -28.7335110, 41.9202194, -30.6917725, 44.5793228, -73.3128357, 72.6119843

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9514561, upper bound: 77.9515722
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9514561, upper bound: 77.9517803
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -34.8573303, 41.6929207, -35.2874756, 42.2161980, -77.0735168, 76.9803925
1: -26.8566914, 33.0270042, -27.1880989, 33.4509544, -60.3076477, 60.2150993
2: -23.3781586, 32.9910622, -23.6696396, 33.4155960, -56.7937546, 56.6606903
3: -32.2049904, 39.5241165, -32.6121635, 40.0253601, -72.2303238, 72.1362762
4: -30.3103943, 44.0107918, -30.6917725, 44.5793228, -74.8897171, 74.7025604

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9517849, upper bound: 77.9521127
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -77.9517849, upper bound: 77.9523250
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -31.1855011, 37.4674988, -26.1117134, 30.4598427, -61.6453438, 63.5792046
1: -24.1073112, 29.6677723, -19.9750519, 24.0622005, -48.1695099, 49.6428223
2: -20.9856720, 29.6444969, -17.3923473, 24.0895405, -45.0752106, 47.0368423
3: -28.8964233, 35.5232506, -23.9274788, 28.7653179, -57.6617393, 59.4507294
4: -27.2243156, 39.5415764, -22.4313278, 32.1443291, -59.3686447, 61.9728889

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9240267, upper bound: 77.9318500
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9240267, upper bound: 77.9324075
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -33.2685127, 39.6535950, -26.1117134, 30.4598427, -63.7283554, 65.7653046
1: -25.6111336, 31.4002991, -19.9750519, 24.0622005, -49.6733284, 51.3753510
2: -22.3033543, 31.3242931, -17.3923473, 24.0895405, -46.3928871, 48.7166405
3: -30.7320042, 37.5634918, -23.9274788, 28.7653179, -59.4973221, 61.4909706
4: -28.8902245, 41.7764015, -22.4313278, 32.1443291, -61.0345421, 64.2077179

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9247777, upper bound: 77.9369971
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -77.9247777, upper bound: 77.9375546
time: 1.15 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.56 + 418.68 = 422.24 seconds
