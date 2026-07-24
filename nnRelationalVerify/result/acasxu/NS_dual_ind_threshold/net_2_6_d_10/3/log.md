## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 23.9931544845


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.5559692, 16.9026623, -9.5559692, 16.9026623, -26.4586315, 26.4586315)
1: (-1.9498287, 2.4651399, -1.9498287, 2.4651399, -4.4149685, 4.4149685)
2: (-1.5350443, 2.0507975, -1.5350443, 2.0507975, -3.5858412, 3.5858412)
3: (-1.4968777, 3.7269382, -1.4968777, 3.7269382, -5.2238150, 5.2238150)
4: (-1.5394063, 2.7020743, -1.5394063, 2.7020743, -4.2414808, 4.2414808)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.36 + 1.29 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24.1137231, upper bound: 24.1137231

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0198988, upper bound: 24.0987607
time: 0.40 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0206071
time: 0.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -24.0198988, upper bound: 24.0987607
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.00
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0206071

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.4695539, 6.7397461, -9.5559692, 16.9026623, -21.3722153, 16.2957153
1: -0.7916995, 1.0438734, -1.9498287, 2.4651399, -3.2568393, 2.9937022
2: -0.6411211, 0.8345411, -1.5350443, 2.0507975, -2.6919186, 2.3695846
3: -0.6209559, 1.5210140, -1.4968777, 3.7269382, -4.3478937, 3.0178916
4: -0.6379681, 1.1251349, -1.5394063, 2.7020743, -3.3400424, 2.6645408

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
time: 0.40 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
time: 0.40 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.0110874, 13.6429243, -9.5559692, 16.9026623, -24.9137478, 23.1988945
1: -1.5030907, 1.9807596, -1.9498287, 2.4651399, -3.9682307, 3.9305882
2: -1.2374809, 1.6453836, -1.5350443, 2.0507975, -3.2882783, 3.1804271
3: -1.1593255, 3.0259297, -1.4968777, 3.7269382, -4.8862638, 4.5228066
4: -1.1869328, 2.1908948, -1.5394063, 2.7020743, -3.8890071, 3.7303004

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0191958
time: 0.39 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0206071
time: 0.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.22 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0206071
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0191958
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0206071

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.4695539, 6.7397461, -4.4695539, 6.7397461, -11.2093000, 11.2093000
1: -0.7916995, 1.0438734, -0.7916995, 1.0438734, -1.8355730, 1.8355730
2: -0.6411211, 0.8345411, -0.6411211, 0.8345411, -1.4756622, 1.4756622
3: -0.6209559, 1.5210140, -0.6209559, 1.5210140, -2.1419699, 2.1419699
4: -0.6379681, 1.1251349, -0.6379681, 1.1251349, -1.7631030, 1.7631030

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0197454, upper bound: 24.0973494
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0198988, upper bound: 24.0368996
time: 0.42 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.4695539, 6.7397461, -8.0110874, 13.6429243, -18.1124783, 14.7508335
1: -0.7916995, 1.0438734, -1.5030907, 1.9807596, -2.7724590, 2.5469642
2: -0.6411211, 0.8345411, -1.2374809, 1.6453836, -2.2865043, 2.0720217
3: -0.6209559, 1.5210140, -1.1593255, 3.0259297, -3.6468856, 2.6803393
4: -0.6379681, 1.1251349, -1.1869328, 2.1908948, -2.8288624, 2.3120677

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0197454, upper bound: 24.0987607
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0198988, upper bound: 24.0383109
time: 0.41 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.0110874, 13.6429243, -4.4695539, 6.7397461, -14.7508335, 18.1124763
1: -1.5030907, 1.9807596, -0.7916995, 1.0438734, -2.5469642, 2.7724590
2: -1.2374809, 1.6453836, -0.6411211, 0.8345411, -2.0720217, 2.2865043
3: -1.1593255, 3.0259297, -0.6209559, 1.5210140, -2.6803393, 3.6468856
4: -1.1869328, 2.1908948, -0.6379681, 1.1251349, -2.3120677, 2.8288624

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0143030, upper bound: 24.0095456
time: 0.41 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0191958
time: 0.45 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.0110874, 13.6429243, -8.0110874, 13.6429243, -21.6540089, 21.6540089
1: -1.5030907, 1.9807596, -1.5030907, 1.9807596, -3.4838505, 3.4838505
2: -1.2374809, 1.6453836, -1.2374809, 1.6453836, -2.8828642, 2.8828642
3: -1.1593255, 3.0259297, -1.1593255, 3.0259297, -4.1852551, 4.1852551
4: -1.1869328, 2.1908948, -1.1869328, 2.1908948, -3.3778276, 3.3778276

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0143030, upper bound: 24.0095456
time: 0.41 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958
time: 0.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.27 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0197454, upper bound: 24.0973494
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0198988, upper bound: 24.0368996
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0197454, upper bound: 24.0987607
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0198988, upper bound: 24.0383109
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0143030, upper bound: 24.0095456
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0206071, upper bound: 24.0191958
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0143030, upper bound: 24.0095456
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -24.0191958, upper bound: 24.0191958

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -4.4695539, 6.7397461, -9.3260536, 6.9784174
1: -0.3424560, 0.4800463, -0.7916995, 1.0438734, -1.3863294, 1.2717459
2: -0.2596823, 0.3596359, -0.6411211, 0.8345411, -1.0942234, 1.0007570
3: -0.2769262, 0.6608293, -0.6209559, 1.5210140, -1.7979400, 1.2817852
4: -0.2771527, 0.4922103, -0.6379681, 1.1251349, -1.4022876, 1.1301782

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0374492, upper bound: 24.0374492
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0374492, upper bound: 24.0376025
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -4.4695539, 6.7397461, -10.8106155, 10.0907373
1: -0.6731225, 0.9151498, -0.7916995, 1.0438734, -1.7169960, 1.7068492
2: -0.5482491, 0.7177234, -0.6411211, 0.8345411, -1.3827901, 1.3588445
3: -0.5329784, 1.3033334, -0.6209559, 1.5210140, -2.0539923, 1.9242892
4: -0.5411942, 0.9680334, -0.6379681, 1.1251349, -1.6663291, 1.6060016

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0376025, upper bound: 24.0374492
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0376025, upper bound: 24.0376025
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -8.0110874, 13.6429243, -16.2292271, 10.5199509
1: -0.3424560, 0.4800463, -1.5030907, 1.9807596, -2.3232155, 1.9831371
2: -0.2596823, 0.3596359, -1.2374809, 1.6453836, -1.9050658, 1.5971168
3: -0.2769262, 0.6608293, -1.1593255, 3.0259297, -3.3028557, 1.8201548
4: -0.2771527, 0.4922103, -1.1869328, 2.1908948, -2.4680476, 1.6791431

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0320068
time: 0.43 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0383109
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -8.0110874, 13.6429243, -17.7137890, 13.6322708
1: -0.6731225, 0.9151498, -1.5030907, 1.9807596, -2.6538820, 2.4182405
2: -0.5482491, 0.7177234, -1.2374809, 1.6453836, -2.1936321, 1.9552042
3: -0.5329784, 1.3033334, -1.1593255, 3.0259297, -3.5589077, 2.4626589
4: -0.5411942, 0.9680334, -1.1869328, 2.1908948, -2.7320890, 2.1549659

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320068
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0383109
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -4.4695539, 6.7397461, -10.8630028, 9.5827904
1: -0.5457485, 0.8533046, -0.7916995, 1.0438734, -1.5896220, 1.6450040
2: -0.4990431, 0.6817743, -0.6411211, 0.8345411, -1.3335842, 1.3228953
3: -0.4728078, 1.2374785, -0.6209559, 1.5210140, -1.9938216, 1.8584344
4: -0.4858359, 0.9272271, -0.6379681, 1.1251349, -1.6109709, 1.5651952

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -4.4695539, 6.7397461, -14.2657223, 16.9392242
1: -1.3758048, 1.8236271, -0.7916995, 1.0438734, -2.4196782, 2.6153266
2: -1.1365285, 1.5210134, -0.6411211, 0.8345411, -1.9710696, 2.1621344
3: -1.0658861, 2.7853112, -0.6209559, 1.5210140, -2.5869002, 3.4062672
4: -1.0868741, 2.0250101, -0.6379681, 1.1251349, -2.2120090, 2.6629777

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0197454
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0198988
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -8.0110874, 13.6429243, -17.7661762, 13.1243248
1: -0.5457485, 0.8533046, -1.5030907, 1.9807596, -2.5265081, 2.3563952
2: -0.4990431, 0.6817743, -1.2374809, 1.6453836, -2.1444266, 1.9192549
3: -0.4728078, 1.2374785, -1.1593255, 3.0259297, -3.4987371, 2.3968039
4: -0.4858359, 0.9272271, -1.1869328, 2.1908948, -2.6767306, 2.1141598

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0095456
time: 0.42 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -8.0110874, 13.6429243, -21.1688976, 20.4807549
1: -1.3758048, 1.8236271, -1.5030907, 1.9807596, -3.3565645, 3.3267176
2: -1.1365285, 1.5210134, -1.2374809, 1.6453836, -2.7819118, 2.7584944
3: -1.0658861, 2.7853112, -1.1593255, 3.0259297, -4.0918159, 3.9446368
4: -1.0868741, 2.0250101, -1.1869328, 2.1908948, -3.2777691, 3.2119429

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0143030
time: 0.44 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0191958
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.32 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0374492, upper bound: 24.0374492
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0374492, upper bound: 24.0376025
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0376025, upper bound: 24.0374492
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0376025, upper bound: 24.0376025
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0320068
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0383109
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320068
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0383109
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0100951
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0320068, upper bound: 24.0102485
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0197454
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0198988
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0046527
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0046527, upper bound: 24.0095456
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0143030
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0191958

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -2.5863075, 2.5088634, -5.0951705, 5.0951705
1: -0.3424560, 0.4800463, -0.3424560, 0.4800463, -0.8225023, 0.8225023
2: -0.2596823, 0.3596359, -0.2596823, 0.3596359, -0.6193182, 0.6193182
3: -0.2769262, 0.6608293, -0.2769262, 0.6608293, -0.9377555, 0.9377555
4: -0.2771527, 0.4922103, -0.2771527, 0.4922103, -0.7693629, 0.7693629

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0349177, upper bound: 24.0973294
time: 0.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0956001
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -4.0708694, 5.6211839, -8.2074909, 6.5797319
1: -0.3424560, 0.4800463, -0.6731225, 0.9151498, -1.2576058, 1.1531688
2: -0.2596823, 0.3596359, -0.5482491, 0.7177234, -0.9774057, 0.9078849
3: -0.2769262, 0.6608293, -0.5329784, 1.3033334, -1.5802596, 1.1938077
4: -0.2771527, 0.4922103, -0.5411942, 0.9680334, -1.2451861, 1.0334044

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0349177, upper bound: 24.0974653
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -2.5863075, 2.5088634, -6.5797319, 8.2074909
1: -0.6731225, 0.9151498, -0.3424560, 0.4800463, -1.1531688, 1.2576057
2: -0.5482491, 0.7177234, -0.2596823, 0.3596359, -0.9078850, 0.9774057
3: -0.5329784, 1.3033334, -0.2769262, 0.6608293, -1.1938077, 1.5802596
4: -0.5411942, 0.9680334, -0.2771527, 0.4922103, -1.0334045, 1.2451861

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0350711, upper bound: 24.0368797
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -4.0708694, 5.6211839, -9.6920519, 9.6920519
1: -0.6731225, 0.9151498, -0.6731225, 0.9151498, -1.5882723, 1.5882723
2: -0.5482491, 0.7177234, -0.5482491, 0.7177234, -1.2659724, 1.2659724
3: -0.5329784, 1.3033334, -0.5329784, 1.3033334, -1.8363118, 1.8363117
4: -0.5411942, 0.9680334, -0.5411942, 0.9680334, -1.5092275, 1.5092275

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0350711, upper bound: 24.0368797
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -4.1232567, 5.1132369, -7.6995444, 6.6321197
1: -0.3424560, 0.4800463, -0.5457485, 0.8533046, -1.1957604, 1.0257949
2: -0.2596823, 0.3596359, -0.4990431, 0.6817743, -0.9414564, 0.8586790
3: -0.2769262, 0.6608293, -0.4728078, 1.2374785, -1.5144048, 1.1336371
4: -0.2771527, 0.4922103, -0.4858359, 0.9272271, -1.2043798, 0.9780462

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0924565
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0095257, upper bound: 24.0907271
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.5863075, 2.5088634, -7.5259757, 12.4696693, -15.0559731, 10.0348396
1: -0.3424560, 0.4800463, -1.3758048, 1.8236271, -2.1660829, 1.8558511
2: -0.2596823, 0.3596359, -1.1365285, 1.5210134, -1.7806956, 1.4961644
3: -0.2769262, 0.6608293, -1.0658861, 2.7853112, -3.0622373, 1.7267153
4: -0.2771527, 0.4922103, -1.0868741, 2.0250101, -2.3021629, 1.5790844

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0987606
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0095257, upper bound: 24.0970313
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -4.1232567, 5.1132369, -9.1841068, 9.7444391
1: -0.6731225, 0.9151498, -0.5457485, 0.8533046, -1.5264270, 1.4608984
2: -0.5482491, 0.7177234, -0.4990431, 0.6817743, -1.2300234, 1.2167665
3: -0.5329784, 1.3033334, -0.4728078, 1.2374785, -1.7704567, 1.7761412
4: -0.5411942, 0.9680334, -0.4858359, 0.9272271, -1.4684212, 1.4538693

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320068
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0096615, upper bound: 24.0294753
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0708694, 5.6211839, -7.5259757, 12.4696693, -16.5405369, 13.1471596
1: -0.6731225, 0.9151498, -1.3758048, 1.8236271, -2.4967496, 2.2909546
2: -0.5482491, 0.7177234, -1.1365285, 1.5210134, -2.0692623, 1.8542519
3: -0.5329784, 1.3033334, -1.0658861, 2.7853112, -3.3182893, 2.3692195
4: -0.5411942, 0.9680334, -1.0868741, 2.0250101, -2.5662041, 2.0549076

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320102
time: 0.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0096615, upper bound: 24.0295915
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -2.5863075, 2.5088634, -6.6321197, 7.6995444
1: -0.5457485, 0.8533046, -0.3424560, 0.4800463, -1.0257949, 1.1957605
2: -0.4990431, 0.6817743, -0.2596823, 0.3596359, -0.8586790, 0.9414566
3: -0.4728078, 1.2374785, -0.2769262, 0.6608293, -1.1336370, 1.5144048
4: -0.4858359, 0.9272271, -0.2771527, 0.4922103, -0.9780462, 1.2043798

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -4.0708694, 5.6211839, -9.7444391, 9.1841068
1: -0.5457485, 0.8533046, -0.6731225, 0.9151498, -1.4608984, 1.5264270
2: -0.4990431, 0.6817743, -0.5482491, 0.7177234, -1.2167665, 1.2300234
3: -0.4728078, 1.2374785, -0.5329784, 1.3033334, -1.7761412, 1.7704569
4: -0.4858359, 0.9272271, -0.5411942, 0.9680334, -1.4538693, 1.4684212

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -2.5863075, 2.5088634, -10.0348396, 15.0559731
1: -1.3758048, 1.8236271, -0.3424560, 0.4800463, -1.8558511, 2.1660829
2: -1.1365285, 1.5210134, -0.2596823, 0.3596359, -1.4961644, 1.7806957
3: -1.0658861, 2.7853112, -0.2769262, 0.6608293, -1.7267154, 3.0622373
4: -1.0868741, 2.0250101, -0.2771527, 0.4922103, -1.5790844, 2.3021629

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0191759
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0185173
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -4.0708694, 5.6211839, -13.1471586, 16.5405350
1: -1.3758048, 1.8236271, -0.6731225, 0.9151498, -2.2909546, 2.4967496
2: -1.1365285, 1.5210134, -0.5482491, 0.7177234, -1.8542519, 2.0692623
3: -1.0658861, 2.7853112, -0.5329784, 1.3033334, -2.3692195, 3.3182893
4: -1.0868741, 2.0250101, -0.5411942, 0.9680334, -2.0549076, 2.5662041

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0363047, upper bound: 23.9403725
time: 0.42 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0198988
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -4.1232567, 5.1132369, -9.2364941, 9.2364941
1: -0.5457485, 0.8533046, -0.5457485, 0.8533046, -1.3990531, 1.3990531
2: -0.4990431, 0.6817743, -0.4990431, 0.6817743, -1.1808174, 1.1808174
3: -0.4728078, 1.2374785, -0.4728078, 1.2374785, -1.7102863, 1.7102863
4: -0.4858359, 0.9272271, -0.4858359, 0.9272271, -1.4130629, 1.4130629

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.1232567, 5.1132369, -7.5259757, 12.4696693, -16.5929241, 12.6392126
1: -0.5457485, 0.8533046, -1.3758048, 1.8236271, -2.3693757, 2.2291093
2: -0.4990431, 0.6817743, -1.1365285, 1.5210134, -2.0200565, 1.8183026
3: -0.4728078, 1.2374785, -1.0658861, 2.7853112, -3.2581189, 2.3033648
4: -0.4858359, 0.9272271, -1.0868741, 2.0250101, -2.5108459, 2.0141013

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -4.1232567, 5.1132369, -12.6392126, 16.5929241
1: -1.3758048, 1.8236271, -0.5457485, 0.8533046, -2.2291093, 2.3693757
2: -1.1365285, 1.5210134, -0.4990431, 0.6817743, -1.8183026, 2.0200565
3: -1.0658861, 2.7853112, -0.4728078, 1.2374785, -2.3033645, 3.2581186
4: -1.0868741, 2.0250101, -0.4858359, 0.9272271, -2.0141013, 2.5108459

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0089506, upper bound: 23.9347767
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0143030
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.5259757, 12.4696693, -7.5259757, 12.4696693, -19.9956417, 19.9956417
1: -1.3758048, 1.8236271, -1.3758048, 1.8236271, -3.1994319, 3.1994319
2: -1.1365285, 1.5210134, -1.1365285, 1.5210134, -2.6575418, 2.6575418
3: -1.0658861, 2.7853112, -1.0658861, 2.7853112, -3.8511972, 3.8511972
4: -1.0868741, 2.0250101, -1.0868741, 2.0250101, -3.1118841, 3.1118841

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0089507, upper bound: 23.9410810
time: 0.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0191958
time: 0.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.45 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0349177, upper bound: 24.0973294
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0956001
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0349177, upper bound: 24.0974653
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0350711, upper bound: 24.0368797
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0350711, upper bound: 24.0368797
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0924565
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0095257, upper bound: 24.0907271
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0100951, upper bound: 24.0987606
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0095257, upper bound: 24.0970313
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320068
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0096615, upper bound: 24.0294753
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0102485, upper bound: 24.0320102
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0096615, upper bound: 24.0295915
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0191759
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0185173
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0363047, upper bound: 23.9403725
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0198988
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0089506, upper bound: 23.9347767
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0143030
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0089507, upper bound: 23.9410810
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 0, lower bound: -24.0109569, upper bound: 24.0191958

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -2.5863075, 2.5088634, -4.4081211, 4.0827885
1: -0.2512228, 0.4153590, -0.3424560, 0.4800463, -0.7312692, 0.7578150
2: -0.1897190, 0.3037605, -0.2596823, 0.3596359, -0.5493550, 0.5634428
3: -0.1865274, 0.5037925, -0.2769262, 0.6608293, -0.8473567, 0.7807187
4: -0.2026858, 0.4101150, -0.2771527, 0.4922103, -0.6948958, 0.6872678

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -2.5863075, 2.5088634, -5.0596843, 5.0247393
1: -0.3349115, 0.4698570, -0.3424560, 0.4800463, -0.8149579, 0.8123129
2: -0.2535889, 0.3495589, -0.2596823, 0.3596359, -0.6132247, 0.6092412
3: -0.2711747, 0.6448022, -0.2769262, 0.6608293, -0.9320040, 0.9217283
4: -0.2704332, 0.4802800, -0.2771527, 0.4922103, -0.7626434, 0.7574327

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -4.0708694, 5.6211839, -7.5204420, 5.5673504
1: -0.2512228, 0.4153590, -0.6731225, 0.9151498, -1.1663727, 1.0884814
2: -0.1897190, 0.3037605, -0.5482491, 0.7177234, -0.9074424, 0.8520095
3: -0.1865274, 0.5037925, -0.5329784, 1.3033334, -1.4898607, 1.0367709
4: -0.2026858, 0.4101150, -0.5411942, 0.9680334, -1.1707191, 0.9513092

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -4.0708694, 5.6211839, -8.1720047, 6.5093007
1: -0.3349115, 0.4698570, -0.6731225, 0.9151498, -1.2500613, 1.1429794
2: -0.2535889, 0.3495589, -0.5482491, 0.7177234, -0.9713122, 0.8978079
3: -0.2711747, 0.6448022, -0.5329784, 1.3033334, -1.5745078, 1.1777805
4: -0.2704332, 0.4802800, -0.5411942, 0.9680334, -1.2384667, 1.0214739

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -2.5863075, 2.5088634, -6.0643158, 6.5806670
1: -0.4660061, 0.7464195, -0.3424560, 0.4800463, -0.9460524, 1.0888754
2: -0.4303486, 0.5797251, -0.2596823, 0.3596359, -0.7899846, 0.8394074
3: -0.3725791, 1.0105162, -0.2769262, 0.6608293, -1.0334084, 1.2874420
4: -0.3907847, 0.8034525, -0.2771527, 0.4922103, -0.8829950, 1.0806053

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0343483
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0957359, upper bound: 24.0343483
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -2.5863075, 2.5088634, -6.5119095, 8.0504742
1: -0.6559309, 0.8941496, -0.3424560, 0.4800463, -1.1359773, 1.2366054
2: -0.5348901, 0.6979164, -0.2596823, 0.3596359, -0.8945261, 0.9575987
3: -0.5199558, 1.2695751, -0.2769262, 0.6608293, -1.1807852, 1.5465013
4: -0.5269762, 0.9435105, -0.2771527, 0.4922103, -1.0191865, 1.2206631

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0957359, upper bound: 24.0343483
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0957359, upper bound: 24.0343483
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -4.0708694, 5.6211839, -9.1766348, 8.0652285
1: -0.4660061, 0.7464195, -0.6731225, 0.9151498, -1.3811557, 1.4195421
2: -0.4303486, 0.5797251, -0.5482491, 0.7177234, -1.1480720, 1.1279743
3: -0.3725791, 1.0105162, -0.5329784, 1.3033334, -1.6759125, 1.5434941
4: -0.3907847, 0.8034525, -0.5411942, 0.9680334, -1.3588182, 1.3446467

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0343483
time: 0.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -4.0708694, 5.6211839, -9.6242285, 9.5350361
1: -0.6559309, 0.8941496, -0.6731225, 0.9151498, -1.5710807, 1.5672721
2: -0.5348901, 0.6979164, -0.5482491, 0.7177234, -1.2526134, 1.2461655
3: -0.5199558, 1.2695751, -0.5329784, 1.3033334, -1.8232892, 1.8025533
4: -0.5269762, 0.9435105, -0.5411942, 0.9680334, -1.4950095, 1.4847047

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0343483
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0343483
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -4.1232567, 5.1132369, -7.0124946, 5.6197376
1: -0.2512228, 0.4153590, -0.5457485, 0.8533046, -1.1045274, 0.9611075
2: -0.1897190, 0.3037605, -0.4990431, 0.6817743, -0.8714932, 0.8028035
3: -0.1865274, 0.5037925, -0.4728078, 1.2374785, -1.4240059, 0.9766003
4: -0.2026858, 0.4101150, -0.4858359, 0.9272271, -1.1299129, 0.8959509

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -4.1232567, 5.1132369, -7.6640577, 6.5616884
1: -0.3349115, 0.4698570, -0.5457485, 0.8533046, -1.1882161, 1.0156054
2: -0.2535889, 0.3495589, -0.4990431, 0.6817743, -0.9353631, 0.8486019
3: -0.2711747, 0.6448022, -0.4728078, 1.2374785, -1.5086530, 1.1176100
4: -0.2704332, 0.4802800, -0.4858359, 0.9272271, -1.1976603, 0.9661157

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -7.5259757, 12.4696693, -14.3689260, 9.0224562
1: -0.2512228, 0.4153590, -1.3758048, 1.8236271, -2.0748498, 1.7911638
2: -0.1897190, 0.3037605, -1.1365285, 1.5210134, -1.7107325, 1.4402890
3: -0.1865274, 0.5037925, -1.0658861, 2.7853112, -2.9718387, 1.5696787
4: -0.2026858, 0.4101150, -1.0868741, 2.0250101, -2.2276959, 1.4969891

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9402191, upper bound: 24.0967544
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0190868, upper bound: 24.0987606
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -7.5259757, 12.4696693, -15.0204859, 9.9644079
1: -0.3349115, 0.4698570, -1.3758048, 1.8236271, -2.1585386, 1.8456618
2: -0.2535889, 0.3495589, -1.1365285, 1.5210134, -1.7746022, 1.4860873
3: -0.2711747, 0.6448022, -1.0658861, 2.7853112, -3.0564859, 1.7106882
4: -0.2704332, 0.4802800, -1.0868741, 2.0250101, -2.2954428, 1.5671539

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9396496, upper bound: 24.0950250
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0185173, upper bound: 24.0970313
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -4.1232567, 5.1132369, -8.6686897, 8.1176167
1: -0.4660061, 0.7464195, -0.5457485, 0.8533046, -1.3193105, 1.2921681
2: -0.4303486, 0.5797251, -0.4990431, 0.6817743, -1.1121229, 1.0787683
3: -0.3725791, 1.0105162, -0.4728078, 1.2374785, -1.6100576, 1.4833233
4: -0.3907847, 0.8034525, -0.4858359, 0.9272271, -1.3180118, 1.2892884

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -4.1232567, 5.1132369, -9.1162834, 9.5874233
1: -0.6559309, 0.8941496, -0.5457485, 0.8533046, -1.5092355, 1.4398981
2: -0.5348901, 0.6979164, -0.4990431, 0.6817743, -1.2166640, 1.1969594
3: -0.5199558, 1.2695751, -0.4728078, 1.2374785, -1.7574342, 1.7423828
4: -0.5269762, 0.9435105, -0.4858359, 0.9272271, -1.4542034, 1.4293464

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -7.5259757, 12.4696693, -16.0251217, 11.5203333
1: -0.4660061, 0.7464195, -1.3758048, 1.8236271, -2.2896333, 2.1222243
2: -0.4303486, 0.5797251, -1.1365285, 1.5210134, -1.9513619, 1.7162535
3: -0.3725791, 1.0105162, -1.0658861, 2.7853112, -3.1578903, 2.0764022
4: -0.3907847, 0.8034525, -1.0868741, 2.0250101, -2.4157946, 1.8903266

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0270961
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0295915
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -7.5259757, 12.4696693, -16.4727154, 12.9901428
1: -0.6559309, 0.8941496, -1.3758048, 1.8236271, -2.4795580, 2.2699542
2: -0.5348901, 0.6979164, -1.1365285, 1.5210134, -2.0559032, 1.8344448
3: -0.5199558, 1.2695751, -1.0658861, 2.7853112, -3.3052671, 2.3354611
4: -0.5269762, 0.9435105, -1.0868741, 2.0250101, -2.5519862, 2.0303845

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0270961
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0295915
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.2360005, 10.6901970, -2.5863075, 2.5088634, -9.7448635, 13.2765045
1: -1.1651489, 1.6386482, -0.3424560, 0.4800463, -1.6451951, 1.9811040
2: -1.0093144, 1.3916780, -0.2596823, 0.3596359, -1.3689504, 1.6513602
3: -0.9413394, 2.4850583, -0.2769262, 0.6608293, -1.6021687, 2.7619841
4: -0.9461204, 1.8441526, -0.2771527, 0.4922103, -1.4383307, 2.1213052

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.2505918, 11.8198490, -2.5863075, 2.5088634, -9.7594547, 14.4061565
1: -1.3047245, 1.7375709, -0.3424560, 0.4800463, -1.7847707, 2.0800269
2: -1.0818779, 1.4490128, -0.2596823, 0.3596359, -1.4415139, 1.7086951
3: -1.0168345, 2.6466234, -0.2769262, 0.6608293, -1.6776638, 2.9235494
4: -1.0355235, 1.9309644, -0.2771527, 0.4922103, -1.5277338, 2.2081170

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.8980193, 13.5696831, -4.0708694, 5.6211839, -13.5192022, 17.6405506
1: -1.4928042, 1.8957810, -0.6731225, 0.9151498, -2.4079540, 2.5689034
2: -1.2113233, 1.5952249, -0.5482491, 0.7177234, -1.9290466, 2.1434734
3: -1.1621414, 2.9961805, -0.5329784, 1.3033334, -2.4654746, 3.5291588
4: -1.1629504, 2.1526532, -0.5411942, 0.9680334, -2.1309834, 2.6938472

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0363047, upper bound: 23.9403725
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0337732, upper bound: 23.9397855
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.1789784, 12.0272818, -4.0708694, 5.6211839, -12.8001623, 16.0981503
1: -1.3228519, 1.7437416, -0.6731225, 0.9151498, -2.2380018, 2.4168642
2: -1.0918704, 1.4474474, -0.5482491, 0.7177234, -1.8095938, 1.9956963
3: -1.0151075, 2.6709991, -0.5329784, 1.3033334, -2.3184407, 3.2039771
4: -1.0409853, 1.9349223, -0.5411942, 0.9680334, -2.0090187, 2.4761167

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0192401
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0186531
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.8980193, 13.5696831, -4.1232567, 5.1132369, -13.0112562, 17.6929359
1: -1.4928042, 1.8957810, -0.5457485, 0.8533046, -2.3461087, 2.4415295
2: -1.2113233, 1.5952249, -0.4990431, 0.6817743, -1.8930975, 2.0942678
3: -1.1621414, 2.9961805, -0.4728078, 1.2374785, -2.3996198, 3.4689884
4: -1.1629504, 2.1526532, -0.4858359, 0.9272271, -2.0901775, 2.6384890

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.1789784, 12.0272818, -4.1232567, 5.1132369, -12.2922153, 16.1505356
1: -1.3228519, 1.7437416, -0.5457485, 0.8533046, -2.1761565, 2.2894902
2: -1.0918704, 1.4474474, -0.4990431, 0.6817743, -1.7736447, 1.9464905
3: -1.0151075, 2.6709991, -0.4728078, 1.2374785, -2.2525856, 3.1438067
4: -1.0409853, 1.9349223, -0.4858359, 0.9272271, -1.9682124, 2.4207582

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.8980193, 13.5696831, -7.5259757, 12.4696693, -20.3676891, 21.0956554
1: -1.4928042, 1.8957810, -1.3758048, 1.8236271, -3.3164313, 3.2715857
2: -1.2113233, 1.5952249, -1.1365285, 1.5210134, -2.7323365, 2.7317533
3: -1.1621414, 2.9961805, -1.0658861, 2.7853112, -3.9474525, 4.0620666
4: -1.1629504, 2.1526532, -1.0868741, 2.0250101, -3.1879604, 3.2395272

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9390746, upper bound: 23.9389452
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9390746, upper bound: 23.9410808
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.1789784, 12.0272818, -7.5259757, 12.4696693, -19.6486473, 19.5532551
1: -1.3228519, 1.7437416, -1.3758048, 1.8236271, -3.1464791, 3.1195464
2: -1.0918704, 1.4474474, -1.1365285, 1.5210134, -2.6128838, 2.5839758
3: -1.0151075, 2.6709991, -1.0658861, 2.7853112, -3.8004181, 3.7368851
4: -1.0409853, 1.9349223, -1.0868741, 2.0250101, -3.0659950, 3.0217965

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -23.9391566, upper bound: 23.9519152
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9391566, upper bound: 24.0191958
time: 0.47 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.49 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0956001
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0957359
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0956001, upper bound: 24.0343483
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0957359, upper bound: 24.0343483
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0957359, upper bound: 24.0343483
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0957359, upper bound: 24.0343483
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0343483
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0344841, upper bound: 24.0343483
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0343483
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0343483, upper bound: 24.0343483
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -23.9402191, upper bound: 24.0967544
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0190868, upper bound: 24.0987606
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -23.9396496, upper bound: 24.0950250
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0185173, upper bound: 24.0970313
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0270961
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0295915
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0270961
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0184422, upper bound: 24.0295915
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0970313, upper bound: 24.0185173
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0363047, upper bound: 23.9403725
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0337732, upper bound: 23.9397855
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0383109, upper bound: 24.0192401
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0186531
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.49
Output dim: 0, lower bound: -23.9390746, upper bound: 23.9389452
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.49
Output dim: 0, lower bound: -23.9390746, upper bound: 23.9410808
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.49
Output dim: 0, lower bound: -23.9391566, upper bound: 23.9519152
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -23.9391566, upper bound: 24.0191958

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -1.8992579, 1.4964807, -3.3957386, 3.3957386
1: -0.2512228, 0.4153590, -0.2512228, 0.4153590, -0.6665819, 0.6665819
2: -0.1897190, 0.3037605, -0.1897190, 0.3037605, -0.4934795, 0.4934795
3: -0.1865274, 0.5037925, -0.1865274, 0.5037925, -0.6903200, 0.6903200
4: -0.2026858, 0.4101150, -0.2026858, 0.4101150, -0.6128008, 0.6128008

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -2.5508208, 2.4384320, -4.3376899, 4.0473013
1: -0.2512228, 0.4153590, -0.3349115, 0.4698570, -0.7210798, 0.7502705
2: -0.1897190, 0.3037605, -0.2535889, 0.3495589, -0.5392779, 0.5573493
3: -0.1865274, 0.5037925, -0.2711747, 0.6448022, -0.8313295, 0.7749672
4: -0.2026858, 0.4101150, -0.2704332, 0.4802800, -0.6829655, 0.6805483

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -1.8992579, 1.4964807, -4.0473013, 4.3376899
1: -0.3349115, 0.4698570, -0.2512228, 0.4153590, -0.7502705, 0.7210798
2: -0.2535889, 0.3495589, -0.1897190, 0.3037605, -0.5573493, 0.5392779
3: -0.2711747, 0.6448022, -0.1865274, 0.5037925, -0.7749672, 0.8313296
4: -0.2704332, 0.4802800, -0.2026858, 0.4101150, -0.6805483, 0.6829656

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0737964, upper bound: 24.0876858
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0955783, upper bound: 24.0955784
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -2.5508208, 2.4384320, -4.9892530, 4.9892521
1: -0.3349115, 0.4698570, -0.3349115, 0.4698570, -0.8047685, 0.8047685
2: -0.2535889, 0.3495589, -0.2535889, 0.3495589, -0.6031477, 0.6031477
3: -0.2711747, 0.6448022, -0.2711747, 0.6448022, -0.9159768, 0.9159768
4: -0.2704332, 0.4802800, -0.2704332, 0.4802800, -0.7507131, 0.7507132

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0737964, upper bound: 24.0876858
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0955783, upper bound: 24.0955784
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -3.5554526, 3.9943595, -5.8936172, 5.0519333
1: -0.2512228, 0.4153590, -0.4660061, 0.7464195, -0.9976424, 0.8813651
2: -0.1897190, 0.3037605, -0.4303486, 0.5797251, -0.7694442, 0.7341091
3: -0.1865274, 0.5037925, -0.3725791, 1.0105162, -1.1970429, 0.8763716
4: -0.2026858, 0.4101150, -0.3907847, 0.8034525, -1.0061382, 0.8008997

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -4.0030460, 5.4641671, -7.3634253, 5.4995270
1: -0.2512228, 0.4153590, -0.6559309, 0.8941496, -1.1453724, 1.0712899
2: -0.1897190, 0.3037605, -0.5348901, 0.6979164, -0.8876355, 0.8386506
3: -0.1865274, 0.5037925, -0.5199558, 1.2695751, -1.4561023, 1.0237484
4: -0.2026858, 0.4101150, -0.5269762, 0.9435105, -1.1461962, 0.9370912

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -3.5554526, 3.9943595, -6.5451798, 5.9938846
1: -0.3349115, 0.4698570, -0.4660061, 0.7464195, -1.0813310, 0.9358630
2: -0.2535889, 0.3495589, -0.4303486, 0.5797251, -0.8333139, 0.7799075
3: -0.2711747, 0.6448022, -0.3725791, 1.0105162, -1.2816901, 1.0173812
4: -0.2704332, 0.4802800, -0.3907847, 0.8034525, -1.0738857, 0.8710647

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0125444, upper bound: 24.0878213
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343264, upper bound: 24.0957138
time: 0.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -4.0030460, 5.4641671, -8.0149879, 6.4414778
1: -0.3349115, 0.4698570, -0.6559309, 0.8941496, -1.2290611, 1.1257880
2: -0.2535889, 0.3495589, -0.5348901, 0.6979164, -0.9515052, 0.8844490
3: -0.2711747, 0.6448022, -0.5199558, 1.2695751, -1.5407497, 1.1647580
4: -0.2704332, 0.4802800, -0.5269762, 0.9435105, -1.2139437, 1.0072560

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0125444, upper bound: 24.0878213
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0343264, upper bound: 24.0957138
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -1.8992579, 1.4964807, -5.0519333, 5.8936176
1: -0.4660061, 0.7464195, -0.2512228, 0.4153590, -0.8813651, 0.9976424
2: -0.4303486, 0.5797251, -0.1897190, 0.3037605, -0.7341091, 0.7694442
3: -0.3725791, 1.0105162, -0.1865274, 0.5037925, -0.8763716, 1.1970428
4: -0.3907847, 0.8034525, -0.2026858, 0.4101150, -0.8008997, 1.0061382

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -2.5508208, 2.4384320, -5.9938846, 6.5451803
1: -0.4660061, 0.7464195, -0.3349115, 0.4698570, -0.9358630, 1.0813310
2: -0.4303486, 0.5797251, -0.2535889, 0.3495589, -0.7799075, 0.8333140
3: -0.3725791, 1.0105162, -0.2711747, 0.6448022, -1.0173812, 1.2816901
4: -0.3907847, 0.8034525, -0.2704332, 0.4802800, -0.8710647, 1.0738857

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -1.8992579, 1.4964807, -5.4995270, 7.3634253
1: -0.6559309, 0.8941496, -0.2512228, 0.4153590, -1.0712899, 1.1453724
2: -0.5348901, 0.6979164, -0.1897190, 0.3037605, -0.8386506, 0.8876355
3: -0.5199558, 1.2695751, -0.1865274, 0.5037925, -1.0237484, 1.4561024
4: -0.5269762, 0.9435105, -0.2026858, 0.4101150, -0.9370912, 1.1461964

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -2.5508208, 2.4384320, -6.4414778, 8.0149879
1: -0.6559309, 0.8941496, -0.3349115, 0.4698570, -1.1257880, 1.2290611
2: -0.5348901, 0.6979164, -0.2535889, 0.3495589, -0.8844490, 0.9515052
3: -0.5199558, 1.2695751, -0.2711747, 0.6448022, -1.1647580, 1.5407495
4: -0.5269762, 0.9435105, -0.2704332, 0.4802800, -1.0072560, 1.2139437

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -3.5554526, 3.9943595, -7.5498114, 7.5498114
1: -0.4660061, 0.7464195, -0.4660061, 0.7464195, -1.2124252, 1.2124255
2: -0.4303486, 0.5797251, -0.4303486, 0.5797251, -1.0100738, 1.0100738
3: -0.3725791, 1.0105162, -0.3725791, 1.0105162, -1.3830950, 1.3830948
4: -0.3907847, 0.8034525, -0.3907847, 0.8034525, -1.1942372, 1.1942372

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -4.0030460, 5.4641671, -9.0196199, 7.9974055
1: -0.4660061, 0.7464195, -0.6559309, 0.8941496, -1.3601555, 1.4023503
2: -0.4303486, 0.5797251, -0.5348901, 0.6979164, -1.1282650, 1.1146152
3: -0.3725791, 1.0105162, -0.5199558, 1.2695751, -1.6421542, 1.5304718
4: -0.3907847, 0.8034525, -0.5269762, 0.9435105, -1.3342953, 1.3304287

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -3.5554526, 3.9943595, -7.9974046, 9.0196199
1: -0.6559309, 0.8941496, -0.4660061, 0.7464195, -1.4023503, 1.3601555
2: -0.5348901, 0.6979164, -0.4303486, 0.5797251, -1.1146152, 1.1282649
3: -0.5199558, 1.2695751, -0.3725791, 1.0105162, -1.5304717, 1.6421542
4: -0.5269762, 0.9435105, -0.3907847, 0.8034525, -1.3304287, 1.3342953

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -4.0030460, 5.4641671, -9.4672127, 9.4672127
1: -0.6559309, 0.8941496, -0.6559309, 0.8941496, -1.5500804, 1.5500805
2: -0.5348901, 0.6979164, -0.5348901, 0.6979164, -1.2328062, 1.2328062
3: -0.5199558, 1.2695751, -0.5199558, 1.2695751, -1.7895306, 1.7895305
4: -0.5269762, 0.9435105, -0.5269762, 0.9435105, -1.4704866, 1.4704866

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -7.8980193, 13.5696831, -15.4689407, 9.3944998
1: -0.2512228, 0.4153590, -1.4928042, 1.8957810, -2.1470034, 1.9081632
2: -0.1897190, 0.3037605, -1.2113233, 1.5952249, -1.7849438, 1.5150838
3: -0.1865274, 0.5037925, -1.1621414, 2.9961805, -3.1827080, 1.6659340
4: -0.2026858, 0.4101150, -1.1629504, 2.1526532, -2.3553391, 1.5730654

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.8992579, 1.4964807, -7.1789784, 12.0272818, -13.9265394, 8.6754589
1: -0.2512228, 0.4153590, -1.3228519, 1.7437416, -1.9949644, 1.7382109
2: -0.1897190, 0.3037605, -1.0918704, 1.4474474, -1.6371665, 1.3956308
3: -0.1865274, 0.5037925, -1.0151075, 2.6709991, -2.8575265, 1.5189000
4: -0.2026858, 0.4101150, -1.0409853, 1.9349223, -2.1376081, 1.4511003

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -7.8980193, 13.5696831, -16.1205006, 10.3364506
1: -0.3349115, 0.4698570, -1.4928042, 1.8957810, -2.2306924, 1.9626611
2: -0.2535889, 0.3495589, -1.2113233, 1.5952249, -1.8488137, 1.5608821
3: -0.2711747, 0.6448022, -1.1621414, 2.9961805, -3.2673552, 1.8069434
4: -0.2704332, 0.4802800, -1.1629504, 2.1526532, -2.4230859, 1.6432301

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9178479, upper bound: 24.0871106
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9396298, upper bound: 24.0950031
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.5508208, 2.4384320, -7.1789784, 12.0272818, -14.5781012, 9.6174107
1: -0.3349115, 0.4698570, -1.3228519, 1.7437416, -2.0786531, 1.7927088
2: -0.2535889, 0.3495589, -1.0918704, 1.4474474, -1.7010362, 1.4414291
3: -0.2711747, 0.6448022, -1.0151075, 2.6709991, -2.9421737, 1.6599097
4: -0.2704332, 0.4802800, -1.0409853, 1.9349223, -2.2053556, 1.5212653

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -23.9967163, upper bound: 24.0891162
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0184983, upper bound: 24.0970088
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -7.2360005, 10.6901970, -14.2456484, 11.2303600
1: -0.4660061, 0.7464195, -1.1651489, 1.6386482, -2.1046538, 1.9115679
2: -0.4303486, 0.5797251, -1.0093144, 1.3916780, -1.8220265, 1.5890396
3: -0.3725791, 1.0105162, -0.9413394, 2.4850583, -2.8576372, 1.9518552
4: -0.3907847, 0.8034525, -0.9461204, 1.8441526, -2.2349372, 1.7495729

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.5554526, 3.9943595, -7.2505918, 11.8198490, -15.3753014, 11.2449512
1: -0.4660061, 0.7464195, -1.3047245, 1.7375709, -2.2035770, 2.0511439
2: -0.4303486, 0.5797251, -1.0818779, 1.4490128, -1.8793614, 1.6616030
3: -0.3725791, 1.0105162, -1.0168345, 2.6466234, -3.0192025, 2.0273502
4: -0.3907847, 0.8034525, -1.0355235, 1.9309644, -2.3217492, 1.8389760

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -7.2360005, 10.6901970, -14.6932430, 12.7001677
1: -0.6559309, 0.8941496, -1.1651489, 1.6386482, -2.2945786, 2.0592980
2: -0.5348901, 0.6979164, -1.0093144, 1.3916780, -1.9265678, 1.7072308
3: -0.5199558, 1.2695751, -0.9413394, 2.4850583, -3.0050139, 2.2109144
4: -0.5269762, 0.9435105, -0.9461204, 1.8441526, -2.3711288, 1.8896309

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0030460, 5.4641671, -7.2505918, 11.8198490, -15.8228941, 12.7147589
1: -0.6559309, 0.8941496, -1.3047245, 1.7375709, -2.3935018, 2.1988735
2: -0.5348901, 0.6979164, -1.0818779, 1.4490128, -1.9839025, 1.7797942
3: -0.5199558, 1.2695751, -1.0168345, 2.6466234, -3.1665792, 2.2864094
4: -0.5269762, 0.9435105, -1.0355235, 1.9309644, -2.4579406, 1.9790341

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.2360005, 10.6901970, -1.8992579, 1.4964807, -8.7324810, 12.5894547
1: -1.1651489, 1.6386482, -0.2512228, 0.4153590, -1.5805079, 1.8898710
2: -1.0093144, 1.3916780, -0.1897190, 0.3037605, -1.3130748, 1.5813971
3: -0.9413394, 2.4850583, -0.1865274, 0.5037925, -1.4451320, 2.6715853
4: -0.9461204, 1.8441526, -0.2026858, 0.4101150, -1.3562354, 2.0468383

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.2360005, 10.6901970, -2.5508208, 2.4384320, -9.6744328, 13.2410164
1: -1.1651489, 1.6386482, -0.3349115, 0.4698570, -1.6350058, 1.9735595
2: -1.0093144, 1.3916780, -0.2535889, 0.3495589, -1.3588730, 1.6452668
3: -0.9413394, 2.4850583, -0.2711747, 0.6448022, -1.5861416, 2.7562330
4: -0.9461204, 1.8441526, -0.2704332, 0.4802800, -1.4264004, 2.1145859

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.2505918, 11.8198490, -1.8992579, 1.4964807, -8.7470722, 13.7191067
1: -1.3047245, 1.7375709, -0.2512228, 0.4153590, -1.7200835, 1.9887937
2: -1.0818779, 1.4490128, -0.1897190, 0.3037605, -1.3856385, 1.6387317
3: -1.0168345, 2.6466234, -0.1865274, 0.5037925, -1.5206270, 2.8331509
4: -1.0355235, 1.9309644, -0.2026858, 0.4101150, -1.4456385, 2.1336501

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.2505918, 11.8198490, -2.5508208, 2.4384320, -9.6890240, 14.3706684
1: -1.3047245, 1.7375709, -0.3349115, 0.4698570, -1.7745814, 2.0724823
2: -1.0818779, 1.4490128, -0.2535889, 0.3495589, -1.4314367, 1.7026017
3: -1.0168345, 2.6466234, -0.2711747, 0.6448022, -1.6616366, 2.9177980
4: -1.0355235, 1.9309644, -0.2704332, 0.4802800, -1.5158035, 2.2013972

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.8980193, 13.5696831, -3.5554526, 3.9943595, -11.8923759, 17.1251354
1: -1.4928042, 1.8957810, -0.4660061, 0.7464195, -2.2392237, 2.3617871
2: -1.2113233, 1.5952249, -0.4303486, 0.5797251, -1.7910484, 2.0255735
3: -1.1621414, 2.9961805, -0.3725791, 1.0105162, -2.1726573, 3.3687596
4: -1.1629504, 2.1526532, -0.3907847, 0.8034525, -1.9664029, 2.5434380

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.8980193, 13.5696831, -4.0030460, 5.4641671, -13.3621864, 17.5727291
1: -1.4928042, 1.8957810, -0.6559309, 0.8941496, -2.3869538, 2.5517118
2: -1.2113233, 1.5952249, -0.5348901, 0.6979164, -1.9092398, 2.1301150
3: -1.1621414, 2.9961805, -0.5199558, 1.2695751, -2.4317162, 3.5161364
4: -1.1629504, 2.1526532, -0.5269762, 0.9435105, -2.1064610, 2.6796293

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.1789784, 12.0272818, -3.5554526, 3.9943595, -11.1733370, 15.5827341
1: -1.3228519, 1.7437416, -0.4660061, 0.7464195, -2.0692716, 2.2097478
2: -1.0918704, 1.4474474, -0.4303486, 0.5797251, -1.6715955, 1.8777959
3: -1.0151075, 2.6709991, -0.3725791, 1.0105162, -2.0256236, 3.0435781
4: -1.0409853, 1.9349223, -0.3907847, 0.8034525, -1.8444378, 2.3257070

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0186531
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0186531
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.1789784, 12.0272818, -4.0030460, 5.4641671, -12.6431456, 16.0303268
1: -1.3228519, 1.7437416, -0.6559309, 0.8941496, -2.2170014, 2.3996725
2: -1.0918704, 1.4474474, -0.5348901, 0.6979164, -1.7897867, 1.9823375
3: -1.0151075, 2.6709991, -0.5199558, 1.2695751, -2.2846823, 3.1909549
4: -1.0409853, 1.9349223, -0.5269762, 0.9435105, -1.9844959, 2.4618986

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0186531
time: 0.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24.0357795, upper bound: 24.0186531
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -7.1789784, 12.0272818, -7.1789784, 12.0272818, -19.2062607, 19.2062607
1: -1.3228519, 1.7437416, -1.3228519, 1.7437416, -3.0665936, 3.0665936
2: -1.0918704, 1.4474474, -1.0918704, 1.4474474, -2.5393176, 2.5393176
3: -1.0151075, 2.6709991, -1.0151075, 2.6709991, -3.6861064, 3.6861060
4: -1.0409853, 1.9349223, -1.0409853, 1.9349223, -2.9759078, 2.9759078

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.65 + 301.02 = 304.66 seconds
