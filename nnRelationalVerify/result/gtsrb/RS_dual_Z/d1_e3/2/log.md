## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 1800 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1026459, 16.1026459)
1: (-4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4232368, 8.4232407)
2: (-5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7034760, 9.7034721)
3: (-13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9608154, 8.9608154)
4: (-7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8247299, 10.8247299)
5: (-9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6185303, 11.6185303)
6: (-25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3741531, 10.3741531)
7: (-9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0258636, 11.0258636)
8: (-15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6228867, 12.6228867)
9: (-11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1615219, 9.1615219)
10: (-6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.4354095, 12.4354095)
11: (-0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8858871, 6.8858910)
12: (-17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2778854, 13.2778854)
13: (-25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5166626, 13.5166626)
14: (-23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.8298950, 17.8298950)
15: (-10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2089844, 9.2089844)
16: (-9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4696732, 7.4696712)
17: (-14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4943466, 14.4943466)
18: (4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3750534, 9.3750534)
19: (0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8193169, 7.8193169)
20: (-0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6290817, 7.6290836)
21: (-0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891)
22: (-0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7161751, 8.7161751)
23: (0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7561913, 7.7561913)
24: (3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5140800, 8.5140800)
25: (2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5245895, 8.5245895)
26: (-4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3423157, 13.3423157)
27: (-3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838)
28: (0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3919983, 10.3919983)
29: (-2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6106796, 8.6106834)
30: (-1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3830910, 8.3830910)
31: (1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3780975, 10.3781013)
32: (-22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.3080673, 10.3080673)
33: (-36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.7489128, 12.7489128)
34: (-28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.3344917, 10.3344917)
35: (-23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4684982, 10.4684982)
36: (-26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.1061325, 13.1061325)
37: (-36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.4090996, 12.4090958)
38: (-26.1904259, -6.1886387, -26.1904259, -6.1886387, -12.0633621, 12.0633621)
39: (-40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.2053986, 13.2054024)
40: (-36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.2068253, 9.2068253)
41: (-24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.6263924, 11.6263962)
42: (-21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7760391, 7.7760391)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.78 + 24.37 = 27.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 18, lower bound: -5.6428935, upper bound: 5.6428935

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1654

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6409013, upper bound: 5.6391298
time: 11.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6391298, upper bound: 5.6409013
time: 6.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.51
Output dim: 18, lower bound: -5.6409013, upper bound: 5.6391298
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.51
Output dim: 18, lower bound: -5.6391298, upper bound: 5.6409013

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1026459, 16.1026382
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4180260, 8.4175529
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7052078, 9.7056389
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9621964, 8.9624596
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8261490, 10.8269958
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6229477, 11.6225281
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3448257, 10.3502197
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0259171, 11.0259323
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6049728, 12.6024094
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1605988, 9.1605530
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.4108582, 12.4072838
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8783340, 6.8750343
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2837753, 13.2838974
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5200119, 13.5253525
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7824936, 17.7744217
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2064514, 9.2043228
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4702034, 7.4701614
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4819870, 14.4801407
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3669052, 9.3664093
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8240356, 7.8237038
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6349640, 7.6345234
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7203827, 8.7194748
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7614555, 7.7590714
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5093155, 8.5084763
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5406380, 8.5365391
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3306274, 13.3268509
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3986282, 10.3972778
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6184578, 8.6151123
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3945656, 8.3916893
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3883858, 10.3874130
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2688179, 10.2761230
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6957550, 12.7068710
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2978134, 10.3041992
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4308853, 10.4363365
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0559540, 13.0636139
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3673477, 12.3727570
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -12.0237427, 12.0302505
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1530991, 13.1664429
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1538925, 9.1622772
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5911942, 11.5977631
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7704201, 7.7716904

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6404504, upper bound: 5.6384588
time: 15.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6402318, upper bound: 5.6386788
time: 11.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1026459, 16.1026382
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4175529, 8.4180260
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7056427, 9.7052078
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9624634, 8.9621964
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8269958, 10.8261490
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6225281, 11.6229477
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3502197, 10.3448257
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0259323, 11.0259171
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6024094, 12.6049652
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1605530, 9.1605988
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.4072876, 12.4108620
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8750343, 6.8783340
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2838974, 13.2837753
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5253525, 13.5200157
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7744217, 17.7824936
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2043228, 9.2064514
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4701614, 7.4702034
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4801407, 14.4819870
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3664093, 9.3669052
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8237038, 7.8240356
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6345215, 7.6349640
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7194748, 8.7203827
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7590714, 7.7614517
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5084763, 8.5093155
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5365372, 8.5406380
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3268509, 13.3306351
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3972778, 10.3986282
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6151123, 8.6184578
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3916893, 8.3945694
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3874168, 10.3883858
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2761230, 10.2688179
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.7068634, 12.6957512
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.3041992, 10.2978134
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4363327, 10.4308815
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0636139, 13.0559540
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3727570, 12.3673477
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -12.0302429, 12.0237465
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1664429, 13.1530991
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1622772, 9.1538925
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5977631, 11.5911903
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7716904, 7.7704201

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6386788, upper bound: 5.6402318
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6384588, upper bound: 5.6404504
time: 16.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 18, lower bound: -5.6404504, upper bound: 5.6384588
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 18, lower bound: -5.6402318, upper bound: 5.6386788
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 18, lower bound: -5.6386788, upper bound: 5.6402318
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.01
Output dim: 18, lower bound: -5.6384588, upper bound: 5.6404504

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080780, 16.1081009
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4147720, 8.4132538
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7197189, 9.7209320
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9419899, 8.9398079
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8179550, 10.8179817
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6163712, 11.6153870
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3423958, 10.3496170
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0241547, 11.0239716
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6035004, 12.5996704
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1267700, 9.1205521
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3733749, 12.3624992
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8530235, 6.8452606
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2831268, 13.2833862
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5317993, 13.5386200
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7596436, 17.7484360
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2096939, 9.2073364
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4555550, 7.4530621
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5214081, 14.5260010
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3598328, 9.3593636
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8195763, 7.8183556
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6361046, 7.6327267
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7127457, 8.7116165
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7577591, 7.7537918
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5034065, 8.5015831
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5654221, 8.5568256
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3370514, 13.3324280
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3987503, 10.3966904
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6194839, 8.6157379
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3773117, 8.3708763
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3876305, 10.3863258
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2659683, 10.2752647
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6423607, 12.6610489
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2649269, 10.2747421
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3989143, 10.4083557
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0167465, 13.0288010
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2905350, 12.3082123
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9931259, 12.0034676
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0933304, 13.1163025
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0903816, 9.1079979
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5465469, 11.5590401
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7861214, 7.7896690

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6291419, upper bound: 5.6377811
time: 12.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6397719, upper bound: 5.6271446
time: 5.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1081085, 16.1080704
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4137344, 8.4142952
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7204971, 9.7201462
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9395409, 8.9422531
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8171310, 10.8188057
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6158066, 11.6159439
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3442192, 10.3477936
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0239563, 11.0241776
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6022339, 12.6009369
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1205978, 9.1267242
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3660736, 12.3698006
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8485603, 6.8497238
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2832718, 13.2832489
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5332794, 13.5371361
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7565079, 17.7515717
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2094612, 9.2075653
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4531021, 7.4555149
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5278549, 14.5195618
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3598595, 9.3593369
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8186836, 7.8192482
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6331673, 7.6356621
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7125244, 8.7118378
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7561760, 7.7553749
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5024223, 8.5025673
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5609245, 8.5613251
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3362045, 13.3332672
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3980408, 10.3974037
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6190834, 8.6161385
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3737526, 8.3744354
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3873024, 10.3866577
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2679596, 10.2732735
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6499290, 12.6534767
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2683563, 10.2713165
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4028969, 10.4043694
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0211334, 13.0244064
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3028030, 12.2959442
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9969635, 11.9996300
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1029587, 13.1066742
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0996094, 9.0987663
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5524673, 11.5531158
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7883987, 7.7873917

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6289173, upper bound: 5.6380010
time: 16.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6395536, upper bound: 5.6273663
time: 12.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080780, 16.1081009
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4142990, 8.4137306
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7201462, 9.7205009
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9422493, 8.9395409
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8188095, 10.8171310
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6159439, 11.6158066
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3477936, 10.3442192
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0241776, 11.0239563
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6009369, 12.6022339
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1267242, 9.1205978
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3697968, 12.3660736
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8497238, 6.8485603
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2832489, 13.2832718
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5371399, 13.5332832
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7515717, 17.7565155
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2075653, 9.2094612
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4555168, 7.4531040
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5195541, 14.5278473
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3593369, 9.3598595
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8192482, 7.8186836
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6356621, 7.6331673
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7118378, 8.7125244
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7553749, 7.7561760
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5025673, 8.5024223
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5613251, 8.5609245
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3332748, 13.3362122
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3973999, 10.3980408
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6161385, 8.6190834
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3744354, 8.3737564
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3866615, 10.3872986
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2732735, 10.2679596
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6534767, 12.6499290
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2713165, 10.2683563
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4043694, 10.4029007
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0244064, 13.0211334
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2959442, 12.3028030
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9996338, 11.9969673
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1066742, 13.1029587
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0987663, 9.0996094
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5531158, 11.5524673
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7873917, 7.7883987

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6273663, upper bound: 5.6395536
time: 10.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6380010, upper bound: 5.6289173
time: 15.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1081085, 16.1080704
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4132614, 8.4147682
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7209320, 9.7197151
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9398079, 8.9419861
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8179855, 10.8179550
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6153870, 11.6163712
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3496170, 10.3423958
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0239716, 11.0241547
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5996704, 12.6035004
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1205521, 9.1267700
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3624954, 12.3733749
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8452606, 6.8530235
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2833862, 13.2831268
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5386200, 13.5317993
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7484360, 17.7596436
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2073364, 9.2096939
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4530640, 7.4555569
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5260010, 14.5214081
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3593636, 9.3598328
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8183556, 7.8195763
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6327286, 7.6361027
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7116165, 8.7127457
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7537918, 7.7577591
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5015831, 8.5034065
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5568275, 8.5654240
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3324280, 13.3370514
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3966904, 10.3987541
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6157341, 8.6194839
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3708763, 8.3773117
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3863258, 10.3876305
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2752609, 10.2659721
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6610451, 12.6423569
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2747421, 10.2649269
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4083595, 10.3989182
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0288010, 13.0167465
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3082123, 12.2905350
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -12.0034637, 11.9931259
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1163025, 13.0933304
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1079979, 9.0903816
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5590439, 11.5465469
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7896690, 7.7861214

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1784

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6271446, upper bound: 5.6397719
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6377811, upper bound: 5.6291419
time: 12.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6291419, upper bound: 5.6377811
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6397719, upper bound: 5.6271446
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6289173, upper bound: 5.6380010
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6395536, upper bound: 5.6273663
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6273663, upper bound: 5.6395536
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6380010, upper bound: 5.6289173
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6271446, upper bound: 5.6397719
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.43
Output dim: 18, lower bound: -5.6377811, upper bound: 5.6291419

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080399, 16.1080322
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4147263, 8.4134254
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7193375, 9.7194862
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9413033, 8.9382286
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8173828, 10.8157806
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6159134, 11.6132889
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3437843, 10.3489952
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0235901, 11.0217438
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6027908, 12.5971756
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1261940, 9.1186600
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3732300, 12.3627472
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8495369, 6.8443031
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2814941, 13.2828751
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5311584, 13.5397110
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7593155, 17.7488861
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2095146, 9.2070580
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4555283, 7.4527302
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5195312, 14.5255585
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3581238, 9.3585892
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8189087, 7.8181381
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6356354, 7.6308537
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7131462, 8.7114487
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7555580, 7.7532349
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5003929, 8.5007820
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5632477, 8.5563183
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3351212, 13.3317032
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3977966, 10.3964195
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6184692, 8.6152611
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3756104, 8.3701477
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3879166, 10.3861122
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2677307, 10.2744331
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6420593, 12.6599693
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2628517, 10.2743225
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3958168, 10.4075851
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0151062, 13.0284119
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2868576, 12.3073311
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9900589, 12.0028687
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0933151, 13.1163292
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0914230, 9.1075249
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5465012, 11.5589981
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7865334, 7.7891426

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6287634, upper bound: 5.6372532
time: 15.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6286128, upper bound: 5.6374038
time: 18.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080093, 16.1080704
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4149399, 8.4132195
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7182693, 9.7205582
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9404106, 8.9391212
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8157501, 10.8174057
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6142654, 11.6149292
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3417740, 10.3510056
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0219269, 11.0234146
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6009979, 12.5989685
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1248817, 9.1199760
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3736267, 12.3623581
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8520660, 6.8417740
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2826157, 13.2817612
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5328827, 13.5379829
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7600937, 17.7481003
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2094193, 9.2071533
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4552231, 7.4530354
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5209732, 14.5241241
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3590584, 9.3576546
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8193626, 7.8176880
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6342278, 7.6322594
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7125778, 8.7120171
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7572021, 7.7515907
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5026093, 8.4985695
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5649147, 8.5546513
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3363266, 13.3304977
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3984833, 10.3957329
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6190071, 8.6147194
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3765831, 8.3691711
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3874207, 10.3866119
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2651443, 10.2770233
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6412811, 12.6607513
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2645111, 10.2726631
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3981514, 10.4052505
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0163574, 13.0271606
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2896576, 12.3045311
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9925232, 12.0004005
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0933609, 13.1162834
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0899086, 9.1090393
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5465012, 11.5589943
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7855988, 7.7900810

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6393785, upper bound: 5.6266143
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6392352, upper bound: 5.6267672
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080704, 16.1080017
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136887, 8.4144630
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7201233, 9.7187042
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9388542, 8.9406738
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8165588, 10.8166046
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6153488, 11.6138458
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3456078, 10.3471718
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0233917, 11.0219498
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6015320, 12.5984344
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1200142, 9.1248322
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3659363, 12.3700485
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8450737, 6.8487663
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2816391, 13.2827377
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5326462, 13.5382233
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7561798, 17.7520142
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2092819, 9.2072906
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4530754, 7.4551830
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5259705, 14.5191193
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3581467, 9.3585625
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8180161, 7.8190346
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6327019, 7.6337891
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7129250, 8.7116699
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7539749, 7.7548218
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4994087, 8.5017662
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5587502, 8.5608177
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3342743, 13.3325500
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3970871, 10.3971329
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6180687, 8.6156616
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3720512, 8.3737068
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3875885, 10.3864441
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2697182, 10.2724495
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6496353, 12.6523972
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2662773, 10.2708969
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3997993, 10.4036026
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0194931, 13.0240173
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2991257, 12.2950630
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9938965, 11.9990273
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1029434, 13.1067085
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1006546, 9.0982933
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5524216, 11.5530739
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7888107, 7.7868690

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6285438, upper bound: 5.6374735
time: 17.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6283912, upper bound: 5.6376238
time: 13.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080399, 16.1080399
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4139023, 8.4142570
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7190552, 9.7197723
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9379692, 8.9415665
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8149338, 10.8182297
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6137085, 11.6154938
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3436012, 10.3491821
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0217285, 11.0236130
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5997391, 12.6002350
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1187019, 9.1261482
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3663254, 12.3696594
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8476028, 6.8462372
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2827606, 13.2816162
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5343704, 13.5364990
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7569580, 17.7512360
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2091866, 9.2073860
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4527702, 7.4554882
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5274124, 14.5176773
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3590851, 9.3576279
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8184700, 7.8185806
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6312943, 7.6351929
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7123566, 8.7122383
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7556190, 7.7531738
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5016251, 8.4995537
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5604172, 8.5591507
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3354797, 13.3313446
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3977661, 10.3964462
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6186104, 8.6151237
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3730278, 8.3727303
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3870850, 10.3869476
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2671318, 10.2750359
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6488495, 12.6531830
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2679367, 10.2692375
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4021339, 10.4012680
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0207443, 13.0227737
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3019257, 12.2922630
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9963684, 11.9965591
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1029892, 13.1066628
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0991402, 9.0998116
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5524292, 11.5530739
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7878723, 7.7878036

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6391645, upper bound: 5.6268362
time: 10.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6390206, upper bound: 5.6269889
time: 12.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080399, 16.1080322
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4142532, 8.4138985
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7197723, 9.7190552
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9415703, 8.9379654
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8182297, 10.8149338
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6154938, 11.6137085
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3491821, 10.3436012
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0236130, 11.0217285
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6002350, 12.5997391
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1261482, 9.1187019
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3696594, 12.3663254
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8462372, 6.8476028
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2816162, 13.2827606
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5364990, 13.5343704
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7512360, 17.7569580
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2073860, 9.2091866
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4554863, 7.4527683
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5176773, 14.5274124
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3576279, 9.3590851
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8185806, 7.8184700
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6351929, 7.6312943
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7122383, 8.7123566
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7531738, 7.7556190
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4995537, 8.5016251
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5591507, 8.5604172
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3313446, 13.3354797
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3964462, 10.3977699
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6151237, 8.6186066
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3727303, 8.3730278
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3869476, 10.3870850
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2750359, 10.2671318
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6531830, 12.6488495
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2692375, 10.2679367
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4012642, 10.4021301
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0227737, 13.0207443
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2922592, 12.3019218
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9965591, 11.9963684
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1066666, 13.1029930
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0998116, 9.0991402
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5530701, 11.5524254
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7878036, 7.7878723

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6269889, upper bound: 5.6390206
time: 12.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6268362, upper bound: 5.6391645
time: 18.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080093, 16.1080704
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4144669, 8.4136925
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7187042, 9.7201233
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9406700, 8.9388580
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8166046, 10.8165588
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6138458, 11.6153488
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3471718, 10.3456078
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0219498, 11.0233917
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5984421, 12.6015320
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1248360, 9.1200180
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3700485, 12.3659325
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8487625, 6.8450737
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2827377, 13.2816391
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5382233, 13.5326462
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7520142, 17.7561798
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2072906, 9.2092819
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4551811, 7.4530735
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5191193, 14.5259705
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3585625, 9.3581467
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8190308, 7.8180161
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6337891, 7.6327000
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7116699, 8.7129250
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7548180, 7.7539749
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5017700, 8.4994087
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5608177, 8.5587502
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3325500, 13.3342743
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3971329, 10.3970833
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6156616, 8.6180687
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3737068, 8.3720512
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3864441, 10.3875885
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2724495, 10.2697182
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6523972, 12.6496353
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2708969, 10.2662773
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4035988, 10.3997993
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0240173, 13.0194931
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2950668, 12.2991219
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9990311, 11.9938965
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1067123, 13.1029396
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0982933, 9.1006546
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5530777, 11.5524216
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7868690, 7.7888107

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6376238, upper bound: 5.6283912
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6374735, upper bound: 5.6285437
time: 7.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080704, 16.1080017
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4132156, 8.4149361
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7205582, 9.7182693
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9391212, 8.9404106
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8174057, 10.8157539
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6149292, 11.6142654
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3510056, 10.3417740
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0234146, 11.0219269
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5989685, 12.6009979
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1199760, 9.1248779
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3623581, 12.3736267
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8417740, 6.8520622
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2817612, 13.2826157
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5379868, 13.5328827
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7481003, 17.7600937
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2071533, 9.2094193
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4530334, 7.4552212
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5241165, 14.5209656
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3576546, 9.3590584
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8176880, 7.8193626
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6322594, 7.6342297
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7120171, 8.7125778
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7515907, 7.7572021
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4985695, 8.5026093
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5546494, 8.5649147
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3304977, 13.3363266
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3957367, 10.3984833
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6147232, 8.6190071
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3691711, 8.3765831
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3866119, 10.3874168
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2770233, 10.2651443
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6607513, 12.6412811
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2726631, 10.2645111
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4052467, 10.3981476
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0271606, 13.0163574
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3045273, 12.2896538
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -12.0003967, 11.9925270
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1162796, 13.0933647
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1090393, 9.0899086
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5589981, 11.5465012
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7900810, 7.7855988

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6267672, upper bound: 5.6392352
time: 12.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6266143, upper bound: 5.6393785
time: 12.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080399, 16.1080399
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4134293, 8.4147301
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7194901, 9.7193413
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9382286, 8.9413033
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8157806, 10.8173790
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6132889, 11.6159134
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3489952, 10.3437843
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0217438, 11.0235901
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5971756, 12.6027908
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1186562, 9.1261902
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3627472, 12.3732338
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8443031, 6.8495369
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2828751, 13.2814941
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5397110, 13.5311584
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7488861, 17.7593079
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2070580, 9.2095108
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4527283, 7.4555264
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5255585, 14.5195312
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3585892, 9.3581238
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8181381, 7.8189087
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6308556, 7.6356335
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7114487, 8.7131462
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7532349, 7.7555580
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5007858, 8.5003929
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5563202, 8.5632477
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3317032, 13.3351212
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3964157, 10.3977966
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6152611, 8.6184692
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3701477, 8.3756065
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3861084, 10.3879204
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2744331, 10.2677307
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6599655, 12.6420631
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2743263, 10.2628517
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4075813, 10.3958130
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0284119, 13.0151062
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3073349, 12.2868538
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -12.0028687, 11.9900589
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1163254, 13.0933189
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1075249, 9.0914230
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5589981, 11.5465012
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7891426, 7.7865334

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1698

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6374038, upper bound: 5.6286128
time: 17.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6372532, upper bound: 5.6287634
time: 13.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 33.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6287634, upper bound: 5.6372532
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6286128, upper bound: 5.6374038
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6393785, upper bound: 5.6266143
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6392352, upper bound: 5.6267672
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6285438, upper bound: 5.6374735
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6283912, upper bound: 5.6376238
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6391645, upper bound: 5.6268362
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6390206, upper bound: 5.6269889
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6269889, upper bound: 5.6390206
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6268362, upper bound: 5.6391645
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6376238, upper bound: 5.6283912
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6374735, upper bound: 5.6285437
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6267672, upper bound: 5.6392352
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6266143, upper bound: 5.6393785
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6374038, upper bound: 5.6286128
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 33.48
Output dim: 18, lower bound: -5.6372532, upper bound: 5.6287634

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080933, 16.1080399
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4139557, 8.4124298
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7193451, 9.7195473
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9395943, 8.9357986
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8179855, 10.8164902
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6159210, 11.6132660
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3437195, 10.3489151
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0227585, 11.0208511
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6028137, 12.5971756
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1206894, 9.1117592
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3659973, 12.3542519
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8494663, 6.8441925
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2804871, 13.2820511
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5256958, 13.5346794
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7522049, 17.7431335
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2085495, 9.2064056
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4498291, 7.4461174
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5036316, 14.5120239
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3602562, 9.3614082
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8184395, 7.8177147
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6353416, 7.6304474
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7108841, 8.7094460
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7563400, 7.7538910
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5016251, 8.5024338
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5612221, 8.5545006
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3322525, 13.3294525
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3986588, 10.3971443
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6157265, 8.6127281
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3749142, 8.3692818
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3876648, 10.3860550
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2683678, 10.2747421
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6403656, 12.6580963
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2632828, 10.2744789
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3960342, 10.4078178
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0082855, 13.0225067
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2814751, 12.3028374
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9807892, 11.9947701
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0915985, 13.1150970
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0915871, 9.1077805
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5468445, 11.5592346
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7843895, 7.7864189

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6114273, upper bound: 5.6368137
time: 10.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6283233, upper bound: 5.6199337
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080475, 16.1080856
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4137344, 8.4126511
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7193985, 9.7194939
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9388771, 8.9365234
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8180923, 10.8163795
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6158905, 11.6132965
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3437042, 10.3489304
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0226974, 11.0209198
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6027985, 12.5971909
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1192932, 9.1131592
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3647385, 12.3555145
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8494282, 6.8442307
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2806625, 13.2818680
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5261307, 13.5342484
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7535629, 17.7417755
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2088585, 9.2060966
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4489136, 7.4470291
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5059891, 14.5096664
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3609428, 9.3607216
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8184853, 7.8176689
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6352272, 7.6305618
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7111435, 8.7091827
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7562103, 7.7540207
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5020409, 8.5020142
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5614319, 8.5542927
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3328705, 13.3288422
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3985214, 10.3972816
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6159363, 8.6125183
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3747463, 8.3694496
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3878632, 10.3858604
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2680359, 10.2750740
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6401901, 12.6582718
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2630043, 10.2747574
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3960495, 10.4078064
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0092087, 13.0215912
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2823601, 12.3019485
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9819641, 11.9935951
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0920868, 13.1146088
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0916786, 9.1076851
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5467377, 11.5593452
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7838097, 7.7869987

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6112679, upper bound: 5.6369643
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6281737, upper bound: 5.6200868
time: 8.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080627, 16.1080780
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4141617, 8.4122238
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7182770, 9.7206154
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9387093, 8.9366913
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8163528, 10.8181152
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6142807, 11.6149139
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3417130, 10.3509254
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0210953, 11.0225143
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6010208, 12.5989685
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1193771, 9.1130753
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3663940, 12.3538628
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8519955, 6.8416634
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2816086, 13.2809219
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5274200, 13.5329552
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7529907, 17.7423553
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2084541, 9.2064972
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4495239, 7.4464226
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5050735, 14.5105896
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3611908, 9.3604698
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8188934, 7.8172607
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6339378, 7.6318531
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7103119, 8.7100143
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7579880, 7.7522430
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5038376, 8.5002174
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5628891, 8.5528336
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3334579, 13.3282471
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3993454, 10.3964577
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6162643, 8.6121864
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3758907, 8.3683052
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3871689, 10.3865547
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2657814, 10.2773285
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6395874, 12.6588783
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2649422, 10.2728157
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3983688, 10.4054832
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0095367, 13.0212631
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2842751, 12.3000374
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9832535, 11.9923019
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0916443, 13.1150513
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0900726, 9.1092949
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5468521, 11.5592346
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7834549, 7.7873573

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6220847, upper bound: 5.6261756
time: 13.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6389372, upper bound: 5.6092688
time: 10.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080017, 16.1081161
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4139404, 8.4124451
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7183304, 9.7205620
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9379768, 8.9374161
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8164673, 10.8180046
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6142502, 11.6149445
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3416977, 10.3509407
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0210266, 11.0225830
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6010056, 12.5989914
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1179810, 9.1144714
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3651276, 12.3551254
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8519535, 6.8417015
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2817841, 13.2807541
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5278549, 13.5325241
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7543488, 17.7409973
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2087669, 9.2061882
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4486084, 7.4473343
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5074387, 14.5082245
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3618774, 9.3597832
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8189354, 7.8172188
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6338234, 7.6319675
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7105751, 8.7097511
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7578583, 7.7523727
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5042572, 8.4997978
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5630989, 8.5526257
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3340759, 13.3276367
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3992081, 10.3965950
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6164742, 8.6119766
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3757229, 8.3684731
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3873596, 10.3863602
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2654495, 10.2776604
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6394119, 12.6590538
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2646637, 10.2730980
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3983841, 10.4054718
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0104523, 13.0203400
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2851601, 12.2991486
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9844284, 11.9911270
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0921326, 13.1145630
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0901642, 9.1092033
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5467453, 11.5593414
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7828712, 7.7879372

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6219329, upper bound: 5.6263284
time: 8.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6387944, upper bound: 5.6094275
time: 9.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1081238, 16.1080093
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4129181, 8.4134674
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7201309, 9.7187614
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9371529, 8.9382439
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8171616, 10.8173141
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6153641, 11.6138306
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3455429, 10.3470917
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0225601, 11.0210495
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6015472, 12.5984421
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1145172, 9.1179352
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3587036, 12.3615532
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8450031, 6.8486519
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2806320, 13.2818985
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5271835, 13.5331955
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7490768, 17.7462692
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2083168, 9.2066383
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4473763, 7.4485703
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5100784, 14.5055847
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3602791, 9.3613815
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8175468, 7.8186073
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6324081, 7.6333828
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7106628, 8.7096672
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7547569, 7.7554741
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5006409, 8.5034180
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5567245, 8.5590019
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3314133, 13.3302994
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3979492, 10.3978539
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6153259, 8.6131287
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3713512, 8.3728409
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3873367, 10.3863869
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2703552, 10.2727547
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6479340, 12.6505280
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2667084, 10.2710533
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4000168, 10.4038315
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0126724, 13.0181198
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2937431, 12.2905693
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9846191, 11.9909286
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1012192, 13.1054764
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1008148, 9.0985489
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5527725, 11.5533142
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7866669, 7.7841454

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6112003, upper bound: 5.6370340
time: 17.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6281046, upper bound: 5.6201561
time: 7.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080780, 16.1080551
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4126968, 8.4136887
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7201843, 9.7187080
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9364281, 8.9389687
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8172684, 10.8172035
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6153336, 11.6138611
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3455276, 10.3471069
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0224915, 11.0211182
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6015320, 12.5984573
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1131210, 9.1193314
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3574371, 12.3628159
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8449650, 6.8486938
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2808075, 13.2817307
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5276108, 13.5327606
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7504349, 17.7449112
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2086258, 9.2063255
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4464607, 7.4494820
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5124359, 14.5032196
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3609657, 9.3606949
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8175888, 7.8185616
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6322937, 7.6334972
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7109222, 8.7094040
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7546272, 7.7556038
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5010567, 8.5029984
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5569305, 8.5587921
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3320236, 13.3296814
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3978043, 10.3979950
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6155357, 8.6129189
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3711834, 8.3730087
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3875275, 10.3861923
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2700233, 10.2730827
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6477585, 12.6507034
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2664299, 10.2713318
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4000320, 10.4038239
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0135956, 13.0171967
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2946281, 12.2896805
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9857941, 11.9897537
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1017075, 13.1049881
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1009102, 9.0984573
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5526657, 11.5534210
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7860832, 7.7847252

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6110398, upper bound: 5.6371836
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6279525, upper bound: 5.6203086
time: 17.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080933, 16.1080475
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4131241, 8.4132614
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7190628, 9.7198334
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9362602, 8.9391365
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8155289, 10.8189392
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6137161, 11.6154709
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3435364, 10.3491020
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0208969, 11.0227127
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5997543, 12.6002350
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1132050, 9.1192474
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3590927, 12.3611641
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8475323, 6.8461266
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2817535, 13.2807846
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5289078, 13.5314674
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7498550, 17.7454910
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2082214, 9.2067299
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4470673, 7.4488754
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5115204, 14.5041428
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3612175, 9.3604469
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8180008, 7.8181534
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6310043, 7.6347866
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7100906, 8.7102356
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7564049, 7.7538300
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5028534, 8.5012016
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5583916, 8.5573330
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3326187, 13.3290939
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3986359, 10.3971710
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6158638, 8.6125870
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3723278, 8.3718643
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3868332, 10.3868866
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2677689, 10.2753410
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6471558, 12.6513062
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2683716, 10.2693901
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4023514, 10.4014969
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0139236, 13.0168686
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2965431, 12.2877693
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9870911, 11.9884605
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1012650, 13.1054230
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0993004, 9.1000633
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5527725, 11.5533104
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7857285, 7.7850800

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6218638, upper bound: 5.6263975
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6387236, upper bound: 5.6094952
time: 14.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080322, 16.1080933
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4129028, 8.4134827
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7191162, 9.7197800
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9355354, 8.9398613
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8156433, 10.8188286
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6136856, 11.6155014
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3435211, 10.3491173
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0208282, 11.0227814
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5997391, 12.6002502
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1118011, 9.1206474
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3578262, 12.3624268
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8474903, 6.8461647
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2819290, 13.2806091
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5293427, 13.5310364
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7512131, 17.7441330
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2085304, 9.2064209
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4461555, 7.4497871
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5138779, 14.5017853
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3619041, 9.3597603
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8180428, 7.8181114
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6308899, 7.6349010
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7103539, 8.7099724
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7562752, 7.7539597
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5032730, 8.5007820
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5585976, 8.5571251
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3332291, 13.3284760
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3984909, 10.3973083
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6160736, 8.6123772
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3721600, 8.3720360
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3870239, 10.3866959
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2674370, 10.2756691
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6469803, 12.6514854
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2680893, 10.2696724
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4023666, 10.4014893
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0148392, 13.0159531
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2974281, 12.2868805
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9882660, 11.9872856
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1017532, 13.1049347
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0993919, 9.0999718
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5526657, 11.5534210
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7851448, 7.7856636

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6217106, upper bound: 5.6265500
time: 8.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6385799, upper bound: 5.6096522
time: 16.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080933, 16.1080399
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4134827, 8.4129028
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7197800, 9.7191124
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9398613, 8.9355354
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8188248, 10.8156433
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6155014, 11.6136856
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3491173, 10.3435211
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0227814, 11.0208282
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6002502, 12.5997391
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1206512, 9.1118050
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3624268, 12.3578300
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8461666, 6.8474922
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2806091, 13.2819290
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5310364, 13.5293388
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7441330, 17.7512131
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2064209, 9.2085304
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4497871, 7.4461555
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5017853, 14.5138779
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3597603, 9.3619041
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8181114, 7.8180428
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6349030, 7.6308899
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7099724, 8.7103539
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7539597, 7.7562714
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5007820, 8.5032730
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5571251, 8.5585976
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3284760, 13.3332367
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3973083, 10.3984947
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6123772, 8.6160736
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3720303, 8.3721581
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3866959, 10.3870277
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2756691, 10.2674370
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6514816, 12.6469803
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2696686, 10.2680893
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4014893, 10.4023628
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0159531, 13.0148392
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2868843, 12.2974319
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9872894, 11.9882698
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1049347, 13.1017532
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0999718, 9.0993919
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5534210, 11.5526657
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7856636, 7.7851448

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6096522, upper bound: 5.6385799
time: 18.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6265500, upper bound: 5.6217106
time: 6.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080475, 16.1080856
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4132614, 8.4131241
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7198334, 9.7190590
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9391365, 8.9362602
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8189392, 10.8155327
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6154709, 11.6137161
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3491020, 10.3435364
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0227127, 11.0208969
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6002350, 12.5997543
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1192474, 9.1132011
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3611603, 12.3590927
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8461285, 6.8475304
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2807846, 13.2817535
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5314713, 13.5289078
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7454910, 17.7498550
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2067299, 9.2082214
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4488754, 7.4470711
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5041428, 14.5115204
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3604469, 9.3612175
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8181534, 7.8179970
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6347885, 7.6310043
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7102356, 8.7100945
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7538261, 7.7564011
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5012016, 8.5028534
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5573349, 8.5583916
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3290939, 13.3326187
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3971710, 10.3986320
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6125870, 8.6158638
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3718624, 8.3723259
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3868866, 10.3868332
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2753410, 10.2677689
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6513062, 12.6471558
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2693901, 10.2683716
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4014969, 10.4023514
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0168686, 13.0139236
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2877693, 12.2965431
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9884644, 11.9870911
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1054230, 13.1012650
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1000633, 9.0993004
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5533142, 11.5527725
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7850800, 7.7857285

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6094952, upper bound: 5.6387236
time: 18.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6263975, upper bound: 5.6218638
time: 5.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080627, 16.1080780
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136887, 8.4126968
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7187119, 9.7201843
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9389687, 8.9364243
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8172073, 10.8172684
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6138611, 11.6153336
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3471069, 10.3455276
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0211182, 11.0224915
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5984573, 12.6015320
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1193314, 9.1131210
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3628159, 12.3574371
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8486919, 6.8449669
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2817307, 13.2808075
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5327606, 13.5276146
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7449112, 17.7504349
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2063255, 9.2086258
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4494820, 7.4464607
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5032272, 14.5124359
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3606949, 9.3609657
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8185616, 7.8175888
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6334953, 7.6322937
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7094040, 8.7109222
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7556038, 7.7546272
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5029984, 8.5010567
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5587921, 8.5569305
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3296814, 13.3320312
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3979950, 10.3978081
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6129189, 8.6155357
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3730068, 8.3711853
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3861923, 10.3875275
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2730827, 10.2700233
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6507034, 12.6477585
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2713318, 10.2664299
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4038239, 10.4000282
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0171967, 13.0135956
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2896843, 12.2946281
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9897537, 11.9858017
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1049881, 13.1017075
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0984573, 9.1009064
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5534210, 11.5526619
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7847252, 7.7860832

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6203086, upper bound: 5.6279525
time: 14.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6371836, upper bound: 5.6110398
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080017, 16.1081161
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4134674, 8.4129181
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7187653, 9.7201309
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9382439, 8.9371529
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8173141, 10.8171577
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6138306, 11.6153641
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3470917, 10.3455429
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0210495, 11.0225601
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5984421, 12.6015472
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1179352, 9.1145172
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3615570, 12.3586998
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8486538, 6.8450050
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2819061, 13.2806320
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5331955, 13.5271835
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7462692, 17.7490692
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2066383, 9.2083168
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4485703, 7.4473763
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5055847, 14.5100784
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3613815, 9.3602791
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8186073, 7.8175468
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6333809, 7.6324081
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7096672, 8.7106590
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7554741, 7.7547569
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5034180, 8.5006371
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5590019, 8.5567245
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3302994, 13.3314133
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3978577, 10.3979454
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6131287, 8.6153259
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3728390, 8.3713531
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3863831, 10.3873367
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2727547, 10.2703552
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6505280, 12.6479378
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2710533, 10.2667084
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4038315, 10.4000168
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0181198, 13.0126724
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2905693, 12.2937393
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9909286, 11.9846230
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1054764, 13.1012192
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0985489, 9.1008148
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5533142, 11.5527687
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7841454, 7.7866669

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6201561, upper bound: 5.6281046
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6370340, upper bound: 5.6112003
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1081238, 16.1080093
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4124451, 8.4139404
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7205658, 9.7183304
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9374123, 8.9379807
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8180008, 10.8164673
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6149445, 11.6142502
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3509407, 10.3416977
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0225830, 11.0210266
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5989914, 12.6010056
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1144714, 9.1179771
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3551254, 12.3651314
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8417034, 6.8519554
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2807541, 13.2817841
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5325241, 13.5278549
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7409973, 17.7543488
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2061882, 9.2087669
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4473343, 7.4486084
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5082245, 14.5074310
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3597832, 9.3618774
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8172150, 7.8189354
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6319695, 7.6338234
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7097549, 8.7105751
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7523766, 7.7578583
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4997978, 8.5042572
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5526276, 8.5630989
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3276367, 13.3340759
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3965988, 10.3992043
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6119766, 8.6164742
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3684750, 8.3757210
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3863602, 10.3873596
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2776604, 10.2654495
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6590576, 12.6394081
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2730980, 10.2646637
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4054718, 10.3983803
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0203400, 13.0104523
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2991524, 12.2851639
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9911270, 11.9844284
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1145630, 13.0921326
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1092033, 9.0901642
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5593414, 11.5467415
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7879372, 7.7828712

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6094275, upper bound: 5.6387944
time: 12.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6263284, upper bound: 5.6219329
time: 13.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080780, 16.1080551
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4122238, 8.4141617
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7206192, 9.7182770
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9366951, 8.9387054
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8181152, 10.8163567
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6149139, 11.6142807
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3509254, 10.3417130
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0225143, 11.0210953
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5989761, 12.6010208
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1130753, 9.1193771
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3538589, 12.3663940
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8416653, 6.8519936
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2809296, 13.2816086
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5329514, 13.5274239
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7423553, 17.7529907
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2064972, 9.2084541
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4464226, 7.4495201
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5105820, 14.5050812
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3604698, 9.3611908
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8172607, 7.8188934
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6318550, 7.6339378
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7100143, 8.7103119
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7522469, 7.7579880
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5002174, 8.5038376
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5528336, 8.5628891
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3282471, 13.3334656
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3964539, 10.3993454
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6121864, 8.6162643
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3683071, 8.3758888
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3865585, 10.3871689
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2773285, 10.2657814
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6588821, 12.6395836
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2728195, 10.2649422
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4054794, 10.3983688
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0212631, 13.0095367
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3000374, 12.2842751
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9923019, 11.9832535
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1150513, 13.0916443
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1092949, 9.0900726
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5592346, 11.5468483
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7873573, 7.7834549

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6092688, upper bound: 5.6389372
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6261756, upper bound: 5.6220847
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080933, 16.1080475
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4126511, 8.4137344
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7194901, 9.7193985
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9365273, 8.9388733
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8163834, 10.8180923
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6132965, 11.6158905
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3489304, 10.3437042
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0209198, 11.0226898
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5971909, 12.6027985
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1131592, 9.1192932
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3555145, 12.3647385
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8442287, 6.8494301
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2818756, 13.2806625
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5342484, 13.5261307
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7417755, 17.7535629
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2060966, 9.2088585
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4470291, 7.4489136
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5096664, 14.5059967
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3607216, 9.3609390
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8176689, 7.8184853
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6305618, 7.6352272
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7091827, 8.7111435
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7540207, 7.7562103
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5020142, 8.5020409
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5542908, 8.5614319
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3288422, 13.3328705
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3972778, 10.3985214
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6125183, 8.6159363
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3694515, 8.3747444
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3858566, 10.3878632
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2750740, 10.2680359
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6582718, 12.6401901
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2747574, 10.2630043
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4078064, 10.3960457
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0215836, 13.0092087
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3019524, 12.2823601
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9935913, 11.9819603
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1146088, 13.0920868
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1076889, 9.0916786
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5593491, 11.5467377
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7869987, 7.7838097

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6200868, upper bound: 5.6281737
time: 9.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6369643, upper bound: 5.6112679
time: 13.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1080322, 16.1080933
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4124298, 8.4139557
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7195435, 9.7193451
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9357948, 8.9395981
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8164902, 10.8179817
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6132660, 11.6159210
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3489151, 10.3437195
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0208511, 11.0227585
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5971756, 12.6028137
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1117554, 9.1206894
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3542557, 12.3660011
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8441906, 6.8494682
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2820435, 13.2804947
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5346756, 13.5256958
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7431412, 17.7522049
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2064056, 9.2085495
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4461174, 7.4498253
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5120316, 14.5036316
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3614082, 9.3602524
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8177147, 7.8184395
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6304474, 7.6353416
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7094460, 8.7108803
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7538910, 7.7563400
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5024338, 8.5016212
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5545006, 8.5612221
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3294525, 13.3322601
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3971405, 10.3986588
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6127281, 8.6157265
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3692837, 8.3749123
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3860550, 10.3876686
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2747421, 10.2683678
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6580963, 12.6403656
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2744789, 10.2632828
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4078140, 10.3960342
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0225067, 13.0082855
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3028374, 12.2814713
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9947662, 11.9807854
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1150970, 13.0915985
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1077805, 9.0915871
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5592346, 11.5468445
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7864189, 7.7843895

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6199337, upper bound: 5.6283233
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6368137, upper bound: 5.6114273
time: 6.11 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6114273, upper bound: 5.6368137
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6283233, upper bound: 5.6199337
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6112679, upper bound: 5.6369643
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6281737, upper bound: 5.6200868
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6220847, upper bound: 5.6261756
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6389372, upper bound: 5.6092688
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6219329, upper bound: 5.6263284
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6387944, upper bound: 5.6094275
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6112003, upper bound: 5.6370340
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6281046, upper bound: 5.6201561
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6110398, upper bound: 5.6371836
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6279525, upper bound: 5.6203086
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6218638, upper bound: 5.6263975
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6387236, upper bound: 5.6094952
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6217106, upper bound: 5.6265500
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6385799, upper bound: 5.6096522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6096522, upper bound: 5.6385799
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6265500, upper bound: 5.6217106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6094952, upper bound: 5.6387236
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6263975, upper bound: 5.6218638
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6203086, upper bound: 5.6279525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6371836, upper bound: 5.6110398
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6201561, upper bound: 5.6281046
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6370340, upper bound: 5.6112003
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6094275, upper bound: 5.6387944
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6263284, upper bound: 5.6219329
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6092688, upper bound: 5.6389372
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6261756, upper bound: 5.6220847
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6200868, upper bound: 5.6281737
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6369643, upper bound: 5.6112679
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6199337, upper bound: 5.6283233
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.69
Output dim: 18, lower bound: -5.6368137, upper bound: 5.6114273

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1053009, 16.1049194
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4138565, 8.4126129
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7165375, 9.7149086
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9392738, 8.9353333
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8137283, 10.8094673
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6153793, 11.6124878
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3360443, 10.3420982
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0215836, 11.0189209
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5990448, 12.5911713
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1154976, 9.1040192
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3659897, 12.3542480
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8408108, 6.8388367
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2754745, 13.2789459
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5255585, 13.5346489
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7497787, 17.7409058
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2045937, 9.2004089
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4481125, 7.4435997
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5027695, 14.5114822
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3581581, 9.3600388
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8182297, 7.8175545
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6352043, 7.6302395
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7114258, 8.7087364
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7545204, 7.7527885
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4965248, 8.4992065
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5570259, 8.5513115
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3319702, 13.3293304
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3973122, 10.3962669
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6137619, 8.6115379
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3667278, 8.3642960
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3881760, 10.3856239
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2686768, 10.2746162
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6389618, 12.6563263
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2540932, 10.2687798
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3920670, 10.4053383
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0044937, 13.0201416
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2808495, 12.3024521
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9699326, 11.9887161
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0895844, 13.1124573
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0912437, 9.1070786
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5439682, 11.5571709
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7803917, 7.7829285

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6110639, upper bound: 5.6366220
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6112312, upper bound: 5.6364390
time: 5.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049652, 16.1052551
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4141388, 8.4123306
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7147064, 9.7167397
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9391289, 8.9354782
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8109512, 10.8122444
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6151428, 11.6127243
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3369446, 10.3412437
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0208359, 11.0196686
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5968094, 12.5934067
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1129494, 9.1065674
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3659973, 12.3542442
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8441143, 6.8355331
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2773895, 13.2770309
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5256653, 13.5345383
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7499771, 17.7407074
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2025528, 9.2024498
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4473114, 7.4444008
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5030823, 14.5111618
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3588867, 9.3593102
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8182793, 7.8175278
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6351318, 7.6303101
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7101746, 8.7099876
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7552414, 7.7520714
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4983978, 8.4973335
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5580330, 8.5503025
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3321304, 13.3291626
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3977852, 10.3957939
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6145401, 8.6107635
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3699245, 8.3610992
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3872375, 10.3865776
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2682419, 10.2750511
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6385956, 12.6566925
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2575836, 10.2652893
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3935623, 10.4038544
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0059204, 13.0187225
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2810936, 12.3022118
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9747314, 11.9839172
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0889511, 13.1130676
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0908852, 9.1074371
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5447845, 11.5563583
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7809029, 7.7824173

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6279540, upper bound: 5.6197414
time: 16.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6281257, upper bound: 5.6195617
time: 16.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052704, 16.1049576
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136353, 8.4128342
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7165909, 9.7148552
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9385490, 8.9360580
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8138428, 10.8093567
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6153488, 11.6125183
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3360291, 10.3421135
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0215149, 11.0189896
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5990295, 12.5911865
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1141014, 9.1054192
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3647308, 12.3555107
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8407688, 6.8388786
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2756500, 13.2787704
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5259857, 13.5342178
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7511368, 17.7395477
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2049026, 9.2000999
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4472008, 7.4445152
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5051270, 14.5091171
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3588448, 9.3593521
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8182755, 7.8175087
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6350899, 7.6303558
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7116890, 8.7084770
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7543907, 7.7529221
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4969444, 8.4987907
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5572319, 8.5511055
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3325806, 13.3287201
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3971672, 10.3964081
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6139717, 8.6113319
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3665600, 8.3644638
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3883743, 10.3854294
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2683449, 10.2749481
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6387863, 12.6565018
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2538147, 10.2690582
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3920822, 10.4053268
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0054169, 13.0192261
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2817345, 12.3015671
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9711075, 11.9875412
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0900726, 13.1119690
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0913353, 9.1069870
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5438614, 11.5572815
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7798119, 7.7835083

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6109011, upper bound: 5.6367674
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6110748, upper bound: 5.6365948
time: 18.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049194, 16.1052933
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4139175, 8.4125519
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7147598, 9.7166862
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9384041, 8.9362030
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8110657, 10.8121338
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6151123, 11.6127548
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3369293, 10.3412590
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0207672, 11.0197372
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5967941, 12.5934219
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1115532, 9.1079636
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3647308, 12.3555069
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8440762, 6.8355713
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2775650, 13.2768555
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5261002, 13.5341034
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7513351, 17.7393494
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2028618, 9.2021408
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4463997, 7.4453163
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5054398, 14.5087967
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3595734, 9.3586235
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8183250, 7.8174820
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6350174, 7.6304245
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7104378, 8.7097282
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7551117, 7.7522011
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4988174, 8.4969177
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5582428, 8.5500946
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3327484, 13.3285522
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3976402, 10.3959312
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6147499, 8.6105537
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3697567, 8.3612671
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3874359, 10.3863831
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2679100, 10.2753830
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6384201, 12.6568680
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2573051, 10.2655678
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3935623, 10.4038429
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0068436, 13.0177994
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2819862, 12.3013229
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9759064, 11.9827423
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0894394, 13.1125793
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0909767, 9.1073456
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5446777, 11.5564690
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7803230, 7.7830009

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6277992, upper bound: 5.6198900
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6279813, upper bound: 5.6197198
time: 12.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052704, 16.1049500
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4140625, 8.4124069
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7154694, 9.7159805
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9383812, 8.9362221
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8121109, 10.8110924
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6137390, 11.6141281
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3340378, 10.3441505
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0199203, 11.0205841
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5972519, 12.5929642
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1141777, 9.1053352
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3663864, 12.3538589
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8433361, 6.8363113
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2765961, 13.2778244
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5272827, 13.5329247
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7505646, 17.7401276
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2044983, 9.2005043
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4478073, 7.4439087
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5042114, 14.5100327
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3590965, 9.3591042
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8187065, 7.8171005
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6338005, 7.6316452
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7108574, 8.7093086
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7561684, 7.7511444
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4987411, 8.4969902
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5586929, 8.5496445
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3331757, 13.3281250
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3979912, 10.3955803
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6143036, 8.6110001
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3677044, 8.3633194
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3876877, 10.3861275
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2660904, 10.2772026
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6381836, 12.6571083
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2557564, 10.2671204
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3944016, 10.4030037
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0057449, 13.0188980
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2836494, 12.2996597
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9723969, 11.9862480
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0896149, 13.1124115
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0897293, 9.1085930
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5439758, 11.5571709
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7794533, 7.7838669

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6217166, upper bound: 5.6259834
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6218874, upper bound: 5.6258030
time: 5.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049347, 16.1052856
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4143448, 8.4121208
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7136383, 9.7178116
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9382439, 8.9363670
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8093338, 10.8138695
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6135025, 11.6143646
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3348923, 10.3432503
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0191727, 11.0213318
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5950165, 12.5951996
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1116371, 9.1078835
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3663864, 12.3538513
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8466396, 6.8330078
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2785034, 13.2759094
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5273895, 13.5328102
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7507553, 17.7399292
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2024612, 9.2025414
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4470062, 7.4447060
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5045242, 14.5097122
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3598213, 9.3583755
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8187332, 7.8170509
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6337280, 7.6317158
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7096062, 8.7105598
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7568855, 7.7504234
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5006142, 8.4951210
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5597000, 8.5486355
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3333359, 13.3279572
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3984718, 10.3951073
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6150780, 8.6102257
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3709011, 8.3601227
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3867416, 10.3870659
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2656555, 10.2776375
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6378174, 12.6574745
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2592468, 10.2636299
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3958893, 10.4015198
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0071716, 13.0174713
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2838860, 12.2994118
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9772034, 11.9814491
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0889969, 13.1130447
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0893707, 9.1089516
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5447845, 11.5563583
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7799644, 7.7833557

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6385655, upper bound: 5.6090754
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6387390, upper bound: 5.6089018
time: 5.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052246, 16.1049957
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4138412, 8.4126282
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7155228, 9.7159271
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9376564, 8.9369507
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8122177, 10.8109818
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6137085, 11.6141586
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3340225, 10.3441658
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0198517, 11.0206604
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5972366, 12.5929794
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1127892, 9.1067314
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3651199, 12.3551178
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8432980, 6.8363495
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2767639, 13.2776489
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5277100, 13.5324936
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7519226, 17.7387695
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2048111, 9.2001915
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4468956, 7.4448204
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5065689, 14.5076752
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3597794, 9.3584175
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8187485, 7.8170586
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6336861, 7.6317596
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7111168, 8.7090454
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7560387, 7.7512779
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4991608, 8.4965744
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5588989, 8.5494366
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3337860, 13.3275146
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3978539, 10.3957214
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6145096, 8.6107903
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3675365, 8.3634872
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3878860, 10.3859329
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2657585, 10.2775345
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6380005, 12.6572838
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2554779, 10.2673988
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3944168, 10.4029922
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0066681, 13.0179749
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2845345, 12.2987709
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9735794, 11.9850731
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0901031, 13.1119232
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0898209, 9.1085014
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5438690, 11.5572777
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7788734, 7.7844505

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6215604, upper bound: 5.6261314
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6217406, upper bound: 5.6259606
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1048889, 16.1053314
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4141235, 8.4123421
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7136917, 9.7177582
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9375114, 8.9370956
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8094406, 10.8137589
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6134720, 11.6143951
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3348770, 10.3432655
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0191040, 11.0214005
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5950012, 12.5952148
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1102409, 9.1092796
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3651276, 12.3551140
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8466015, 6.8330460
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2786789, 13.2757416
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5278244, 13.5323792
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7521210, 17.7385712
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2027702, 9.2022324
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4460945, 7.4456215
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5068893, 14.5073624
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3605080, 9.3576889
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8187752, 7.8170052
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6336136, 7.6318302
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7098656, 8.7102966
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7567558, 7.7505531
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5010338, 8.4947014
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5599098, 8.5484276
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3339539, 13.3273468
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3983269, 10.3952484
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6152878, 8.6100159
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3707333, 8.3602905
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3869324, 10.3868713
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2653236, 10.2779694
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6376343, 12.6576500
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2589645, 10.2639084
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3958969, 10.4015083
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0080872, 13.0165482
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2847786, 12.2985229
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9783783, 11.9802742
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0894852, 13.1125565
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0894623, 9.1088600
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5446777, 11.5564651
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7793808, 7.7839394

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6384161, upper bound: 5.6092312
time: 15.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6386036, upper bound: 5.6090638
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1053314, 16.1048889
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4128189, 8.4136505
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7173233, 9.7141266
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9368248, 8.9377785
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8129044, 10.8102913
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6148224, 11.6130447
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3378677, 10.3402748
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0213852, 11.0191269
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5977783, 12.5924377
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1093254, 9.1101952
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3586884, 12.3615494
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8363476, 6.8432999
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2756195, 13.2788010
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5270386, 13.5331650
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7466431, 17.7440414
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2043610, 9.2006416
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4456596, 7.4460564
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5092087, 14.5050354
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3581848, 9.3600159
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8173370, 7.8184471
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6322708, 7.6331749
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7112045, 8.7089577
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7529373, 7.7543755
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4955406, 8.5001907
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5525246, 8.5558128
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3311234, 13.3301697
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3965950, 10.3969803
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6133614, 8.6119423
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3631725, 8.3678513
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3878479, 10.3859596
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2706642, 10.2726288
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6465302, 12.6487541
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2575226, 10.2653542
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3960571, 10.4013557
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0088882, 13.0157547
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2931175, 12.2901878
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9737701, 11.9848785
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0992126, 13.1028290
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1004753, 9.0978470
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5498962, 11.5512505
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7826691, 7.7806511

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6108374, upper bound: 5.6368424
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6110043, upper bound: 5.6366584
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049957, 16.1052246
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4131012, 8.4133682
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7154922, 9.7159576
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9366875, 8.9379196
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8101349, 10.8130646
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6145782, 11.6132812
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3387680, 10.3394165
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0206375, 11.0198669
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5955429, 12.5946732
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1067772, 9.1127396
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3586960, 12.3615456
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8396511, 6.8399963
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2775345, 13.2768860
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5271530, 13.5330505
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7468414, 17.7438431
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2023201, 9.2026825
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4448586, 7.4468536
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5095291, 14.5047150
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3589134, 9.3592873
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8173866, 7.8184204
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6321983, 7.6332455
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7099533, 8.7102089
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7536583, 7.7536545
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4974136, 8.4983177
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5535355, 8.5548019
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3312912, 13.3300095
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3970680, 10.3965034
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6141396, 8.6111641
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3663616, 8.3646545
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3869095, 10.3869095
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2702293, 10.2730637
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6461639, 12.6491241
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2610092, 10.2618637
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3975449, 10.3998718
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0103073, 13.0143280
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2933617, 12.2899437
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9785690, 11.9800797
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0985794, 13.1034393
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1001167, 9.0982056
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5507050, 11.5504379
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7831802, 7.7801437

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6277357, upper bound: 5.6199639
time: 18.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6279071, upper bound: 5.6197835
time: 6.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052856, 16.1049347
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4125900, 8.4138718
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7173767, 9.7140732
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9361076, 8.9385033
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8130188, 10.8101807
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6147919, 11.6130753
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3378525, 10.3402901
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0213089, 11.0191956
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5977631, 12.5924530
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1079216, 9.1115913
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3574295, 12.3628120
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8363056, 6.8433380
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2757874, 13.2786255
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5274734, 13.5327301
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7480011, 17.7426834
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2046700, 9.2003326
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4447479, 7.4469681
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5115662, 14.5026779
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3588676, 9.3593292
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8173790, 7.8184013
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6321564, 7.6332893
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7114677, 8.7086983
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7528076, 7.7545052
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4959602, 8.4997749
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5527344, 8.5556030
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3317413, 13.3295593
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3964577, 10.3971176
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6135712, 8.6117287
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3629971, 8.3680191
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3880386, 10.3857651
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2703323, 10.2729607
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6463547, 12.6489296
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2572403, 10.2656326
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3960648, 10.4013443
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0098038, 13.0148315
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2940025, 12.2892990
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9749451, 11.9836998
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0997009, 13.1023407
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1005669, 9.0977554
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5497818, 11.5513573
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7820854, 7.7812347

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6106739, upper bound: 5.6369862
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6108468, upper bound: 5.6368132
time: 11.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049500, 16.1052704
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4128799, 8.4135895
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7155457, 9.7159042
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9359550, 8.9386482
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8102417, 10.8129578
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6145477, 11.6133118
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3387527, 10.3394318
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0205688, 11.0199356
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5955276, 12.5946884
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1053810, 9.1141396
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3574371, 12.3628082
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8396130, 6.8400345
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2777023, 13.2767181
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5275879, 13.5326195
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7481995, 17.7424850
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2026291, 9.2023735
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4439468, 7.4477654
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5118866, 14.5023575
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3596001, 9.3586006
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8174286, 7.8183784
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6320839, 7.6333599
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7102165, 8.7099495
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7535286, 7.7537842
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4978333, 8.4979019
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5537415, 8.5545940
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3319016, 13.3293915
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3969307, 10.3966446
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6143494, 8.6109543
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3661938, 8.3648224
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3871002, 10.3867149
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2698975, 10.2733955
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6459885, 12.6492996
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2607307, 10.2621422
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3975525, 10.3998604
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0112305, 13.0134125
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2942543, 12.2890549
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9797440, 11.9789047
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0990677, 13.1029510
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1002083, 9.0981140
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5505981, 11.5505447
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7825966, 7.7807274

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6275799, upper bound: 5.6201117
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6277602, upper bound: 5.6199415
time: 5.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1053009, 16.1049271
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4130249, 8.4134445
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7162552, 9.7151947
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9359398, 8.9386673
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8112869, 10.8119164
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6131744, 11.6146927
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3358612, 10.3423271
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0197144, 11.0207901
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5959854, 12.5942307
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1080132, 9.1115074
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3590851, 12.3611603
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8388729, 6.8407745
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2767334, 13.2776794
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5287628, 13.5314369
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7474289, 17.7432556
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2042694, 9.2007332
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4453545, 7.4463615
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5106506, 14.5035934
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3591194, 9.3590775
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8178139, 7.8179932
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6308670, 7.6345787
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7106361, 8.7095299
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7545815, 7.7527275
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4977570, 8.4979744
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5541916, 8.5541458
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3323288, 13.3289642
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3972816, 10.3962936
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6138992, 8.6114006
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3641415, 8.3668747
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3873596, 10.3864594
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2680779, 10.2752151
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6457520, 12.6495361
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2591820, 10.2636948
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3983841, 10.3990211
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0101318, 13.0145111
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2959175, 12.2873917
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9762421, 11.9824066
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0992355, 13.1027832
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0989609, 9.0993614
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5498962, 11.5512466
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7817307, 7.7815933

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6214966, upper bound: 5.6262053
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6216668, upper bound: 5.6260246
time: 12.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049652, 16.1052628
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4133072, 8.4131622
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7144241, 9.7170258
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9357948, 8.9388123
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8085098, 10.8146896
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6129379, 11.6149292
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3367157, 10.3414268
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0189743, 11.0215378
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5937500, 12.5964661
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1054649, 9.1140556
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3590851, 12.3611526
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8421764, 6.8374710
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2786484, 13.2757721
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5288773, 13.5313263
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7476273, 17.7430649
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2022247, 9.2027779
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4445534, 7.4471588
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5109711, 14.5032730
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3598480, 9.3583488
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8178406, 7.8179436
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6307945, 7.6346493
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7093849, 8.7107773
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7553062, 7.7520103
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4996300, 8.4961052
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5552025, 8.5531349
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3324966, 13.3288040
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3977547, 10.3958206
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6146774, 8.6106262
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3673382, 8.3636780
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3864059, 10.3873978
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2676430, 10.2756500
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6453857, 12.6499062
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2626724, 10.2602005
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3998795, 10.3975372
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0115585, 13.0130844
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2961540, 12.2871437
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9810410, 11.9776115
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0986252, 13.1034164
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0985985, 9.0997238
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5507050, 11.5504341
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7822380, 7.7810822

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6383523, upper bound: 5.6093020
time: 14.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6385256, upper bound: 5.6091270
time: 16.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052551, 16.1049652
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4128036, 8.4136658
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7163086, 9.7151413
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9352074, 8.9393959
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8113937, 10.8118057
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6131439, 11.6147232
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3358459, 10.3423424
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0196457, 11.0208588
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5959702, 12.5942459
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1066093, 9.1129074
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3578186, 12.3624191
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8388348, 6.8408127
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2769089, 13.2775116
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5291977, 13.5310059
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7487869, 17.7418976
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2045784, 9.2004242
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4444389, 7.4472733
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5130081, 14.5012283
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3598061, 9.3583908
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8178558, 7.8179512
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6307526, 7.6346931
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7108994, 8.7092667
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7544518, 7.7528572
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4981766, 8.4975586
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5544014, 8.5539360
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3329468, 13.3283539
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3971443, 10.3964310
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6141129, 8.6111908
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3639736, 8.3670425
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3875504, 10.3862648
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2677460, 10.2755470
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6455765, 12.6497116
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2589035, 10.2639732
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3983994, 10.3990097
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0110550, 13.0135880
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2968025, 12.2865028
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9774170, 11.9812317
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0997238, 13.1022949
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0990524, 9.0992699
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5497894, 11.5513535
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7811470, 7.7821770

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6213389, upper bound: 5.6263531
time: 9.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6215185, upper bound: 5.6261822
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049194, 16.1053009
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4130859, 8.4133797
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7144775, 9.7169724
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9350700, 8.9395409
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8086166, 10.8145828
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6129074, 11.6149597
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3367004, 10.3414421
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0188980, 11.0216064
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5937347, 12.5964813
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1040611, 9.1154518
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3578262, 12.3624153
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8421383, 6.8375092
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2788239, 13.2755890
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5293121, 13.5308914
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7489853, 17.7416992
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2025375, 9.2024651
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4436417, 7.4480705
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5133286, 14.5009155
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3605347, 9.3576622
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8178825, 7.8179016
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6306801, 7.6347637
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7096481, 8.7105179
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7551727, 7.7521400
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5000496, 8.4956856
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5554085, 8.5529270
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3331070, 13.3281860
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3976173, 10.3959579
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6148872, 8.6104164
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3671703, 8.3638496
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3865967, 10.3872032
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2673111, 10.2759819
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6452103, 12.6500816
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2623940, 10.2604828
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3998795, 10.3975258
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0124741, 13.0121613
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2970467, 12.2862549
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9822159, 11.9764328
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0991135, 13.1029282
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0986900, 9.0996284
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5505981, 11.5505447
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7816544, 7.7816620

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6382026, upper bound: 5.6094553
time: 12.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6383890, upper bound: 5.6092880
time: 13.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1053009, 16.1049194
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4133835, 8.4130859
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7169724, 9.7144775
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9395409, 8.9350662
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8145828, 10.8086205
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6149597, 11.6129074
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3414421, 10.3367004
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0216064, 11.0189056
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5964813, 12.5937347
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1154518, 9.1040649
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3624191, 12.3578262
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8375072, 6.8421402
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2755966, 13.2788239
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5308914, 13.5293121
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7416992, 17.7489853
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2024651, 9.2025375
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4480705, 7.4436417
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5009155, 14.5133286
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3576622, 9.3605347
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8179016, 7.8178825
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6347656, 7.6306801
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7105179, 8.7096481
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7521400, 7.7551765
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4956856, 8.5000496
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5529289, 8.5554104
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3281860, 13.3331070
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3959618, 10.3976173
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6104164, 8.6148872
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3638515, 8.3671722
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3872070, 10.3866005
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2759819, 10.2673111
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6500854, 12.6452065
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2604828, 10.2623940
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3975296, 10.3998833
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0121613, 13.0124817
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2862587, 12.2970467
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9764328, 11.9822159
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1029282, 13.0991135
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0996284, 9.0986900
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5505447, 11.5505981
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7816658, 7.7816544

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6092880, upper bound: 5.6383890
time: 11.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6094553, upper bound: 5.6382026
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049652, 16.1052551
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136658, 8.4128036
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7151413, 9.7163086
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9393959, 8.9352112
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8118057, 10.8113937
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6147232, 11.6131439
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3423424, 10.3358459
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0208588, 11.0196457
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5942459, 12.5959702
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1129112, 9.1066093
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3624191, 12.3578186
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8408146, 6.8388329
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2775116, 13.2769165
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5310059, 13.5291977
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7418976, 17.7487869
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2004242, 9.2045784
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4472733, 7.4444427
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5012283, 14.5130081
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3583908, 9.3598061
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8179512, 7.8178558
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6346931, 7.6307526
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7092667, 8.7108994
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7528572, 7.7544518
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4975586, 8.4981766
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5539360, 8.5543995
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3283539, 13.3329468
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3964348, 10.3971443
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6111908, 8.6141090
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3670406, 8.3639755
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3862686, 10.3875504
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2755470, 10.2677460
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6497116, 12.6455765
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2639732, 10.2589035
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3990097, 10.3983994
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0135880, 13.0110550
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2865028, 12.2968063
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9812317, 11.9774170
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1022949, 13.0997238
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0992699, 9.0990524
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5513535, 11.5497894
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7821770, 7.7811508

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6261822, upper bound: 5.6215185
time: 16.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6263531, upper bound: 5.6213389
time: 5.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052704, 16.1049576
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4131622, 8.4133072
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7170258, 9.7144241
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9388084, 8.9357948
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8146896, 10.8085098
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6149292, 11.6129379
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3414268, 10.3367157
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0215378, 11.0189667
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5964661, 12.5937500
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1140556, 9.1054611
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3611526, 12.3590851
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8374691, 6.8421783
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2757645, 13.2786560
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5313263, 13.5288773
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7430649, 17.7476273
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2027740, 9.2022285
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4471588, 7.4445534
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5032730, 14.5109711
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3583488, 9.3598480
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8179436, 7.8178368
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6346512, 7.6307964
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7107773, 8.7093849
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7520103, 7.7553062
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4961052, 8.4996300
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5531349, 8.5552025
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3288040, 13.3324966
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3958168, 10.3977585
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6106262, 8.6146774
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3636837, 8.3673401
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3873978, 10.3864059
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2756500, 10.2676430
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6499023, 12.6453819
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2602005, 10.2626724
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3975372, 10.3998718
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0130844, 13.0115585
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2871437, 12.2961578
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9776154, 11.9810410
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1034164, 13.0986252
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0997238, 9.0985985
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5504379, 11.5507088
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7810822, 7.7822380

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6091270, upper bound: 5.6385256
time: 14.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6093020, upper bound: 5.6383523
time: 12.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049347, 16.1052933
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4134445, 8.4130249
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7151947, 9.7162552
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9386711, 8.9359360
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8119125, 10.8112869
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6146927, 11.6131744
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3423271, 10.3358612
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0207901, 11.0197144
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5942307, 12.5959854
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1115074, 9.1080093
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3611603, 12.3590813
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8407726, 6.8388748
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2776794, 13.2767334
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5314407, 13.5287666
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7432556, 17.7474289
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2007370, 9.2042656
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4463577, 7.4453545
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5035934, 14.5106506
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3590775, 9.3591194
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8179932, 7.8178139
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6345787, 7.6308670
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7095299, 8.7106361
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7527275, 7.7545815
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4979744, 8.4977570
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5541458, 8.5541916
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3289642, 13.3323288
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3962898, 10.3972816
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6114006, 8.6139030
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3668728, 8.3641434
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3864594, 10.3873558
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2752151, 10.2680779
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6495361, 12.6457520
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2636948, 10.2591820
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3990250, 10.3983879
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0145111, 13.0101318
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2873878, 12.2959175
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9824066, 11.9762421
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1027832, 13.0992355
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0993614, 9.0989609
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5512466, 11.5498962
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7815933, 7.7817307

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6260246, upper bound: 5.6216668
time: 10.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6262053, upper bound: 5.6214966
time: 10.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052704, 16.1049500
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4135895, 8.4128799
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7159042, 9.7155457
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9386482, 8.9359589
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8129578, 10.8102455
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6133118, 11.6145477
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3394318, 10.3387527
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0199356, 11.0205688
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5946884, 12.5955276
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1141396, 9.1053810
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3628082, 12.3574333
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8400364, 6.8396111
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2767105, 13.2777100
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5326157, 13.5275841
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7424850, 17.7481995
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2023735, 9.2026291
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4477654, 7.4439468
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5023575, 14.5118866
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3586006, 9.3596001
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8183746, 7.8174286
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6333580, 7.6320858
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7099495, 8.7102165
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7537842, 7.7535286
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4979019, 8.4978333
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5545921, 8.5537415
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3293915, 13.3319016
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3966408, 10.3969307
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6109543, 8.6143494
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3648281, 8.3661957
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3867188, 10.3871002
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2733955, 10.2698975
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6492996, 12.6459885
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2621422, 10.2607307
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3998642, 10.3975525
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0134125, 13.0112305
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2890587, 12.2942505
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9789047, 11.9797440
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1029587, 13.0990677
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0981140, 9.1002083
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5505447, 11.5505981
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7807274, 7.7825966

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6199415, upper bound: 5.6277602
time: 18.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6201117, upper bound: 5.6275799
time: 11.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049347, 16.1052856
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4138718, 8.4125938
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7140732, 9.7173767
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9385033, 8.9361038
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8101807, 10.8130188
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6130753, 11.6147919
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3402901, 10.3378525
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0191956, 11.0213165
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5924530, 12.5977631
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1115913, 9.1079254
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3628082, 12.3574295
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8433399, 6.8363075
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2786255, 13.2757874
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5327301, 13.5274734
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7426834, 17.7480087
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2003326, 9.2046700
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4469681, 7.4447479
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5026779, 14.5115662
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3593292, 9.3588676
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8184013, 7.8173790
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6332893, 7.6321564
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7086983, 8.7114677
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7545052, 7.7528076
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4997749, 8.4959602
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5556030, 8.5527325
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3295593, 13.3317413
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3971214, 10.3964577
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6117325, 8.6135712
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3680172, 8.3629990
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3857651, 10.3880386
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2729607, 10.2703323
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6489334, 12.6463585
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2656326, 10.2572403
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4013443, 10.3960648
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0148392, 13.0098038
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2892952, 12.2940025
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9837036, 11.9749489
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1023407, 13.0997009
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0977554, 9.1005669
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5513535, 11.5497856
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7812347, 7.7820854

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6368132, upper bound: 5.6108468
time: 11.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6369862, upper bound: 5.6106739
time: 18.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052246, 16.1049957
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4133682, 8.4131012
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7159576, 9.7154922
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9379234, 8.9366837
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8130646, 10.8101349
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6132812, 11.6145782
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3394165, 10.3387680
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0198746, 11.0206375
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5946732, 12.5955429
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1127434, 9.1067772
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3615417, 12.3586960
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8399944, 6.8396530
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2768860, 13.2775269
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5330505, 13.5271530
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7438431, 17.7468414
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2026825, 9.2023201
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4468536, 7.4448586
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5047150, 14.5095291
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3592873, 9.3589134
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8184204, 7.8173866
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6332436, 7.6322002
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7102089, 8.7099533
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7536545, 7.7536583
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4983177, 8.4974136
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5548019, 8.5535355
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3300095, 13.3312912
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3965034, 10.3970718
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6111641, 8.6141396
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3646526, 8.3663635
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3869095, 10.3869057
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2730637, 10.2702293
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6491241, 12.6461639
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2618637, 10.2610092
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.3998718, 10.3975410
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0143280, 13.0103073
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2899437, 12.2933617
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9800797, 11.9785690
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1034470, 13.0985794
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0982056, 9.1001167
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5504379, 11.5507050
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7801437, 7.7831802

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6197835, upper bound: 5.6279071
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6199639, upper bound: 5.6277357
time: 6.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1048889, 16.1053314
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136505, 8.4128151
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7141266, 9.7173233
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9377785, 8.9368286
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8102951, 10.8129120
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6130447, 11.6148224
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3402748, 10.3378677
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0191269, 11.0213776
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5924377, 12.5977783
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1101952, 9.1093254
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3615494, 12.3586922
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8433018, 6.8363457
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2788010, 13.2756195
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5331650, 13.5270386
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7440414, 17.7466431
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2006416, 9.2043610
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4460526, 7.4456596
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5050354, 14.5092087
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3600159, 9.3581848
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8184471, 7.8173370
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6331749, 7.6322708
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7089577, 8.7112045
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7543755, 7.7529373
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5001907, 8.4955406
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5558128, 8.5525246
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3301697, 13.3311234
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3969765, 10.3965988
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6119385, 8.6133614
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3678493, 8.3631668
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3859558, 10.3878479
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2726288, 10.2706642
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6487579, 12.6465340
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2653542, 10.2575226
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4013596, 10.3960571
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0157547, 13.0088882
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2901878, 12.2931137
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9848785, 11.9737740
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1028290, 13.0992126
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0978470, 9.1004753
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5512466, 11.5498924
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7806511, 7.7826691

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6366584, upper bound: 5.6110043
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6368424, upper bound: 5.6108374
time: 8.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1053314, 16.1048889
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4123459, 8.4141235
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7177582, 9.7136917
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9370918, 8.9375153
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8137589, 10.8094406
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6143951, 11.6134720
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3432655, 10.3348770
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0214005, 11.0191040
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5952225, 12.5949936
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1092796, 9.1102371
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3551178, 12.3651237
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8330441, 6.8466034
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2757339, 13.2786865
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5323792, 13.5278244
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7385712, 17.7521210
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2022324, 9.2027702
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4456215, 7.4460945
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5073547, 14.5068893
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3576889, 9.3605080
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8170052, 7.8187752
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6318283, 7.6336155
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7102966, 8.7098656
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7505531, 7.7567558
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4947014, 8.5010338
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5484276, 8.5599098
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3273468, 13.3339539
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3952446, 10.3983307
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6100159, 8.6152878
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3602886, 8.3707275
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3868713, 10.3869324
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2779694, 10.2653236
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6576538, 12.6376381
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2639084, 10.2589645
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4015121, 10.3959007
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0165482, 13.0080872
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2985268, 12.2847786
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9802704, 11.9783745
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1125565, 13.0894852
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1088600, 9.0894623
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5564651, 11.5446777
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7839394, 7.7793808

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6090638, upper bound: 5.6386036
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6092312, upper bound: 5.6384161
time: 7.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049957, 16.1052246
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4126282, 8.4138412
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7159271, 9.7155228
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9369469, 8.9376564
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8109818, 10.8122177
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6141586, 11.6137085
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3441658, 10.3340225
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0206604, 11.0198517
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5929794, 12.5972366
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1067314, 9.1127853
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3551178, 12.3651199
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8363514, 6.8432961
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2776489, 13.2767639
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5324936, 13.5277100
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7387695, 17.7519226
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2001915, 9.2048111
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4448204, 7.4468956
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5076752, 14.5065689
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3584175, 9.3597794
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8170547, 7.8187485
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6317596, 7.6336861
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7090454, 8.7111168
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7512741, 7.7560387
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4965744, 8.4991608
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5494385, 8.5588989
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3275146, 13.3337860
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3957176, 10.3978539
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6107903, 8.6145096
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3634853, 8.3675346
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3859329, 10.3878860
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2775345, 10.2657585
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6572876, 12.6380043
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2673988, 10.2554779
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4029922, 10.3944168
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0179749, 13.0066681
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2987709, 12.2845383
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9850769, 11.9735794
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1119232, 13.0901031
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1085014, 9.0898209
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5572739, 11.5438652
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7844505, 7.7788734

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6259606, upper bound: 5.6217406
time: 10.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6261314, upper bound: 5.6215604
time: 9.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052856, 16.1049347
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4121170, 8.4143448
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7178116, 9.7136383
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9363670, 8.9382401
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8138657, 10.8093338
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6143646, 11.6135025
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3432503, 10.3348923
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0213318, 11.0191727
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5952072, 12.5950089
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1078835, 9.1116371
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3538513, 12.3663864
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8330059, 6.8466415
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2759094, 13.2785034
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5328064, 13.5273933
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7399292, 17.7507553
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2025452, 9.2024574
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4447060, 7.4470062
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5097198, 14.5045242
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3583755, 9.3598213
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8170509, 7.8187332
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6317139, 7.6337299
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7105598, 8.7096062
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7504234, 7.7568855
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4951210, 8.5006142
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5486336, 8.5597019
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3279572, 13.3333359
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3951073, 10.3984680
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6102257, 8.6150780
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3601208, 8.3708992
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3870621, 10.3867378
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2776375, 10.2656555
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6574783, 12.6378136
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2636299, 10.2592468
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4015198, 10.3958893
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0174713, 13.0071716
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2994118, 12.2838898
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9814453, 11.9771996
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1130447, 13.0889969
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1089516, 9.0893707
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5563583, 11.5447845
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7833557, 7.7799606

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6089018, upper bound: 5.6387390
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6090754, upper bound: 5.6385655
time: 19.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049500, 16.1052704
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4124069, 8.4140625
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7159805, 9.7154694
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9362221, 8.9383850
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8110886, 10.8121071
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6141281, 11.6137390
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3441505, 10.3340378
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0205917, 11.0199203
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5929642, 12.5972519
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1053352, 9.1141815
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3538589, 12.3663826
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8363132, 6.8433342
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2778244, 13.2765961
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5329285, 13.5272789
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7401276, 17.7505646
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2005005, 9.2045021
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4439049, 7.4478073
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5100327, 14.5042038
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3591042, 9.3590965
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8171005, 7.8187065
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6316452, 7.6338005
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7093086, 8.7108574
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7511444, 7.7561684
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4969902, 8.4987411
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5496445, 8.5586929
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3281250, 13.3331757
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3955803, 10.3979950
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6110001, 8.6142998
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3633175, 8.3677063
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3861237, 10.3876915
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2772026, 10.2660904
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6571045, 12.6381798
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2671204, 10.2557564
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4030075, 10.3944054
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0188980, 13.0057449
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2996559, 12.2836494
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9862518, 11.9724007
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1124115, 13.0896149
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1085930, 9.0897293
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5571671, 11.5439720
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7838707, 7.7794571

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6258030, upper bound: 5.6218874
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6259834, upper bound: 5.6217166
time: 21.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1053009, 16.1049194
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4125519, 8.4139175
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7166901, 9.7147636
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9361992, 8.9384041
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8121338, 10.8110657
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6127548, 11.6151123
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3412552, 10.3369293
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0197372, 11.0207672
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5934296, 12.5967865
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1079674, 9.1115532
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3555069, 12.3647346
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8355732, 6.8440742
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2768555, 13.2775574
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5341034, 13.5261002
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7393494, 17.7513351
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2021408, 9.2028618
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4453163, 7.4463997
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5087967, 14.5054398
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3586235, 9.3595734
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8174820, 7.8183250
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6304245, 7.6350193
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7097282, 8.7104378
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7522011, 7.7551117
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4969177, 8.4988174
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5500946, 8.5582428
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3285522, 13.3327484
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3959312, 10.3976440
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6105537, 8.6147499
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3612652, 8.3697548
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3863831, 10.3874321
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2753830, 10.2679100
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6568680, 12.6384201
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2655678, 10.2573051
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4038467, 10.3935661
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0177994, 13.0068436
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3013268, 12.2819824
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9827423, 11.9759064
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1125793, 13.0894394
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1073456, 9.0909767
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5564651, 11.5446739
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7830009, 7.7803230

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6197198, upper bound: 5.6279813
time: 12.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6198900, upper bound: 5.6277992
time: 10.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049652, 16.1052628
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4128342, 8.4136353
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7148514, 9.7165947
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9360619, 8.9385490
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8093567, 10.8138428
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6125183, 11.6153488
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3421135, 10.3360291
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0189896, 11.0215149
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5911865, 12.5990295
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1054192, 9.1141014
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3555069, 12.3647308
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8388767, 6.8407707
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2787704, 13.2756500
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5342178, 13.5259857
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7395477, 17.7511368
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2000999, 9.2049026
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4445152, 7.4472008
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5091171, 14.5051193
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3593521, 9.3588448
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8175087, 7.8182716
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6303558, 7.6350899
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7084770, 8.7116890
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7529221, 7.7543907
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4987907, 8.4969444
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5511055, 8.5572319
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3287201, 13.3325806
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3964043, 10.3971710
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6113281, 8.6139717
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3644619, 8.3665619
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3854294, 10.3883743
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2749481, 10.2683449
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6565018, 12.6387863
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2690582, 10.2538147
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4053268, 10.3920822
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0192261, 13.0054169
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3015633, 12.2817345
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9875412, 11.9711075
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1119690, 13.0900726
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1069870, 9.0913353
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5572815, 11.5438614
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7835083, 7.7798119

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6365948, upper bound: 5.6110748
time: 22.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6367674, upper bound: 5.6109011
time: 7.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1052551, 16.1049652
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4123306, 8.4141388
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7167435, 9.7147102
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9354744, 8.9391327
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8122406, 10.8109589
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6127243, 11.6151428
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3412399, 10.3369446
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0196686, 11.0208359
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5934067, 12.5968018
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1065636, 9.1129494
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3542480, 12.3659973
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8355350, 6.8441124
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2770309, 13.2773895
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5345383, 13.5256653
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7407074, 17.7499771
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2024498, 9.2025528
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4444008, 7.4473114
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5111618, 14.5030823
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3593102, 9.3588867
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8175278, 7.8182793
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6303101, 7.6351357
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7099876, 8.7101746
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7520714, 7.7552414
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4973335, 8.4983978
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5503044, 8.5580349
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3291626, 13.3321304
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3957939, 10.3977852
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6107635, 8.6145363
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3610973, 8.3699226
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3865738, 10.3872414
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2750511, 10.2682419
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6566925, 12.6385956
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2652893, 10.2575836
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4038544, 10.3935547
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0187225, 13.0059204
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3022118, 12.2810936
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9839172, 11.9747314
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1130676, 13.0889511
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1074371, 9.0908852
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5563583, 11.5447807
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7824211, 7.7809067

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6195617, upper bound: 5.6281257
time: 10.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6197414, upper bound: 5.6279540
time: 14.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1049194, 16.1053009
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4126129, 8.4138565
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7149124, 9.7165413
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9353294, 8.9392776
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8094711, 10.8137321
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6124878, 11.6153793
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3420982, 10.3360443
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0189209, 11.0215836
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5911713, 12.5990448
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1040154, 9.1154976
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3542480, 12.3659935
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8388386, 6.8408089
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2789459, 13.2754745
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5346527, 13.5255547
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7409058, 17.7497787
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2004089, 9.2045937
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4435997, 7.4481125
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.5114822, 14.5027695
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3600388, 9.3581581
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8175545, 7.8182297
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6302414, 7.6352043
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7087364, 8.7114258
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7527924, 7.7545204
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.4992065, 8.4965248
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5513115, 8.5570240
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3293304, 13.3319702
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3962669, 10.3973083
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6115417, 8.6137619
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3642941, 8.3667297
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3856277, 10.3881798
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2746162, 10.2686768
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6563263, 12.6389618
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2687798, 10.2540932
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4053421, 10.3920708
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0201416, 13.0045013
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.3024559, 12.2808456
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9887161, 11.9699326
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.1124573, 13.0895844
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.1070786, 9.0912437
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5571747, 11.5439682
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7829247, 7.7803917

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6364390, upper bound: 5.6112312
time: 10.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6366220, upper bound: 5.6110639
time: 13.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6110639, upper bound: 5.6366220
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6112312, upper bound: 5.6364390
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6279540, upper bound: 5.6197414
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6281257, upper bound: 5.6195617
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6109011, upper bound: 5.6367674
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6110748, upper bound: 5.6365948
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6277992, upper bound: 5.6198900
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6279813, upper bound: 5.6197198
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6217166, upper bound: 5.6259834
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6218874, upper bound: 5.6258030
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6385655, upper bound: 5.6090754
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6387390, upper bound: 5.6089018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6215604, upper bound: 5.6261314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6217406, upper bound: 5.6259606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6384161, upper bound: 5.6092312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6386036, upper bound: 5.6090638
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6108374, upper bound: 5.6368424
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6110043, upper bound: 5.6366584
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6277357, upper bound: 5.6199639
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6279071, upper bound: 5.6197835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6106739, upper bound: 5.6369862
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6108468, upper bound: 5.6368132
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6275799, upper bound: 5.6201117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6277602, upper bound: 5.6199415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6214966, upper bound: 5.6262053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6216668, upper bound: 5.6260246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6383523, upper bound: 5.6093020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6385256, upper bound: 5.6091270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6213389, upper bound: 5.6263531
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6215185, upper bound: 5.6261822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6382026, upper bound: 5.6094553
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6383890, upper bound: 5.6092880
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6092880, upper bound: 5.6383890
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6094553, upper bound: 5.6382026
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6261822, upper bound: 5.6215185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6263531, upper bound: 5.6213389
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6091270, upper bound: 5.6385256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6093020, upper bound: 5.6383523
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6260246, upper bound: 5.6216668
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6262053, upper bound: 5.6214966
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6199415, upper bound: 5.6277602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6201117, upper bound: 5.6275799
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6368132, upper bound: 5.6108468
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6369862, upper bound: 5.6106739
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6197835, upper bound: 5.6279071
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6199639, upper bound: 5.6277357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6366584, upper bound: 5.6110043
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6368424, upper bound: 5.6108374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6090638, upper bound: 5.6386036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6092312, upper bound: 5.6384161
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6259606, upper bound: 5.6217406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6261314, upper bound: 5.6215604
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6089018, upper bound: 5.6387390
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6090754, upper bound: 5.6385655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6258030, upper bound: 5.6218874
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6259834, upper bound: 5.6217166
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6197198, upper bound: 5.6279813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6198900, upper bound: 5.6277992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6365948, upper bound: 5.6110748
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6367674, upper bound: 5.6109011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6195617, upper bound: 5.6281257
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6197414, upper bound: 5.6279540
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6364390, upper bound: 5.6112312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.34
Output dim: 18, lower bound: -5.6366220, upper bound: 5.6110639

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1042023, 16.1039352
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4134064, 8.4120789
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7169647, 9.7154503
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9398270, 8.9345055
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8131714, 10.8089256
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6155167, 11.6125793
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3299904, 10.3342896
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0201111, 11.0169449
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6012344, 12.5941772
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1148300, 9.1019249
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3612289, 12.3493118
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8409195, 6.8389473
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2730560, 13.2761688
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5330124, 13.5413399
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7366333, 17.7302704
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1979294, 9.1955681
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4448433, 7.4395275
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4827957, 14.4944916
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3654518, 9.3691025
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8157120, 7.8155060
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6351700, 7.6298485
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7058983, 8.7040367
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7483063, 7.7474899
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5014687, 8.5059395
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5498276, 8.5458412
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3303146, 13.3295746
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3940392, 10.3933067
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6044083, 8.6035423
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3671684, 8.3646889
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3877144, 10.3859711
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2759476, 10.2792282
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6252556, 12.6401596
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2568588, 10.2694283
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4036331, 10.4150925
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0095062, 13.0247154
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2794151, 12.3015060
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9745331, 11.9928856
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0814056, 13.1027031
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0876617, 9.1035614
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5475769, 11.5592575
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7770233, 7.7787476

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6080444, upper bound: 5.6344492
time: 13.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6090274, upper bound: 5.6334336
time: 6.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1043243, 16.1038055
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4133224, 8.4121666
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7170792, 9.7153358
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9384537, 8.9358788
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8131943, 10.8089027
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6154785, 11.6126251
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3282356, 10.3360443
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0196075, 11.0174332
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6020432, 12.5933609
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1133957, 9.1033516
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3610535, 12.3494797
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8409195, 6.8389473
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2726974, 13.2764893
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5322418, 13.5421066
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7391434, 17.7277603
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1997528, 9.1937447
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4440422, 7.4403324
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4857788, 14.4915161
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3672180, 9.3673325
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8161774, 7.8150330
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6348152, 7.6301956
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7067261, 8.7032127
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7491913, 7.7465744
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5032578, 8.5041504
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5515518, 8.5441132
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3322144, 13.3276825
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3943520, 10.3929977
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6057663, 8.6021881
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3671265, 8.3647346
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3885231, 10.3851585
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2732887, 10.2818871
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6227989, 12.6426163
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2547417, 10.2715454
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4018250, 10.4168968
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0090637, 13.0251427
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2799034, 12.3010216
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9740982, 11.9932823
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0798340, 13.1042747
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0877304, 9.1034927
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5460510, 11.5607758
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7762146, 7.7795296

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6082126, upper bound: 5.6342939
time: 21.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6091703, upper bound: 5.6332386
time: 14.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1038666, 16.1042709
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136887, 8.4117928
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7151337, 9.7172813
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9396744, 8.9346504
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8103943, 10.8117027
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6152802, 11.6128159
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3308907, 10.3334351
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0193634, 11.0176926
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5989990, 12.5964127
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1122818, 9.1044693
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3612289, 12.3493080
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8442230, 6.8356438
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2749710, 13.2742615
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5331268, 13.5412292
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7368240, 17.7300720
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1958885, 9.1976089
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4440422, 7.4403286
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4831161, 14.4941711
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3661804, 9.3683739
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8157616, 7.8154793
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6351013, 7.6299210
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7046471, 8.7052879
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7490273, 7.7467728
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5033417, 8.5040665
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5508347, 8.5448322
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3304825, 13.3294067
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3945122, 10.3928299
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6051865, 8.6027679
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3703651, 8.3614960
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3867760, 10.3869247
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2755127, 10.2796631
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6248817, 12.6405296
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2603493, 10.2659378
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4051132, 10.4136086
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0109406, 13.0232887
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2796593, 12.3012657
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9793320, 11.9880867
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0807800, 13.1033134
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0873032, 9.1039238
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5483856, 11.5584450
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7775345, 7.7782402

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6248907, upper bound: 5.6176099
time: 25.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6258632, upper bound: 5.6165805
time: 22.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1039886, 16.1041412
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4135971, 8.4118805
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7152481, 9.7171669
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9383011, 8.9360237
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8104172, 10.8116798
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6152344, 11.6128616
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3291359, 10.3351898
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0188599, 11.0181808
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5998077, 12.5955963
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1108475, 9.1058998
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3610611, 12.3494797
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8442230, 6.8356438
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2746124, 13.2745667
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5323563, 13.5419922
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7393341, 17.7275620
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1977119, 9.1957855
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4432411, 7.4411297
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4860992, 14.4911957
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3679504, 9.3666039
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8162270, 7.8150063
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6347427, 7.6302662
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7054749, 8.7044640
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7499123, 7.7458572
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5051308, 8.5022774
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5525627, 8.5431042
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3323746, 13.3275146
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3948250, 10.3925209
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6065407, 8.6014099
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3703232, 8.3615379
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3875847, 10.3861122
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2728539, 10.2823219
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6224327, 12.6429825
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2582321, 10.2680550
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4033127, 10.4154091
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0104980, 13.0237160
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2801476, 12.3007812
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9788971, 11.9884834
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0791931, 13.1048927
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0873718, 9.1038551
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5468674, 11.5599670
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7767258, 7.7790222

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6250599, upper bound: 5.6174615
time: 11.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6260097, upper bound: 5.6163903
time: 14.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1041412, 16.1039734
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4131851, 8.4123001
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7170181, 9.7153969
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9390945, 8.9352303
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8132782, 10.8088150
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6154861, 11.6126099
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3299751, 10.3343048
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0200272, 11.0170135
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6012192, 12.5941849
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1134262, 9.1033211
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3599625, 12.3505745
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8408813, 6.8389854
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2731857, 13.2760010
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5334396, 13.5409088
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7379913, 17.7289047
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1982384, 9.1952591
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4439278, 7.4404430
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4851532, 14.4921341
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3661385, 9.3684158
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8157539, 7.8154564
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6350441, 7.6299629
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7061615, 8.7037773
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7481766, 7.7475891
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5018883, 8.5055237
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5500336, 8.5456333
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3309326, 13.3289642
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3939018, 10.3934441
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6046181, 8.6033325
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3670006, 8.3648605
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3879051, 10.3857803
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2756157, 10.2795601
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6250801, 12.6403389
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2565804, 10.2697067
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4036407, 10.4150810
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0104218, 13.0237923
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2803001, 12.3006210
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9756699, 11.9917107
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0818939, 13.1022148
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0877533, 9.1034698
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5474701, 11.5593643
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7764130, 7.7793312

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6077501, upper bound: 5.6346429
time: 11.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6088196, upper bound: 5.6336959
time: 8.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1042633, 16.1038513
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4130936, 8.4123878
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7171326, 9.7152824
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9377213, 8.9366035
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8133087, 10.8087921
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6154480, 11.6126480
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3282204, 10.3360596
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0195389, 11.0175171
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6020355, 12.5933762
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1120071, 9.1047478
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3597946, 12.3507423
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8408813, 6.8389893
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2728729, 13.2763519
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5326767, 13.5416718
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7405014, 17.7263947
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.2000618, 9.1934357
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4431267, 7.4412441
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4881439, 14.4891510
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3679047, 9.3666458
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8162270, 7.8149910
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6347008, 7.6303196
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7069893, 8.7029495
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7490921, 7.7467041
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5036774, 8.5037308
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5517616, 8.5439072
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3328247, 13.3270645
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3942070, 10.3931351
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6059761, 8.6019745
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3669586, 8.3649025
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3887215, 10.3849640
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2729568, 10.2822189
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6226234, 12.6427917
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2544632, 10.2718239
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4018326, 10.4168854
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0099792, 13.0242348
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2807884, 12.3001328
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9752731, 11.9921417
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0803223, 13.1037865
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0878220, 9.1034012
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5459442, 11.5608864
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7756348, 7.7801399

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6079385, upper bound: 5.6344984
time: 15.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6089675, upper bound: 5.6335263
time: 5.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1038055, 16.1043091
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4134750, 8.4120140
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7151871, 9.7172279
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9389572, 8.9353752
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8105011, 10.8115921
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6152496, 11.6128464
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3308754, 10.3334503
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0192795, 11.0177612
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5989838, 12.5964203
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1108856, 9.1058693
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3599701, 12.3505707
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8441849, 6.8356819
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2751007, 13.2740860
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5335541, 13.5407944
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7381821, 17.7287140
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1962013, 9.1972961
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4431267, 7.4412441
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4854736, 14.4918213
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3668671, 9.3676872
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8158035, 7.8154297
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6349754, 7.6300354
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7049103, 8.7050285
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7488976, 7.7468719
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5037613, 8.5036507
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5510445, 8.5446243
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3310928, 13.3287964
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3943748, 10.3929710
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6053963, 8.6025581
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3701935, 8.3616638
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3869667, 10.3867302
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2751808, 10.2799950
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6247063, 12.6407051
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2600708, 10.2662163
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4051285, 10.4135971
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0118408, 13.0223656
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2805519, 12.3003769
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9804764, 11.9869118
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0812683, 13.1028252
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0873947, 9.1038322
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5482788, 11.5585518
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7769241, 7.7788239

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6246025, upper bound: 5.6178122
time: 30.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6256610, upper bound: 5.6168541
time: 13.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1039276, 16.1041870
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4133835, 8.4121017
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7153015, 9.7171135
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9375763, 8.9367485
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8105316, 10.8115692
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6152039, 11.6128922
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3291206, 10.3352051
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0187912, 11.0182648
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5998001, 12.5956116
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1094589, 9.1072960
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3597946, 12.3507385
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8441849, 6.8356819
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2747879, 13.2744370
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5327911, 13.5415611
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7406921, 17.7262039
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1980209, 9.1954765
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4423256, 7.4420452
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4884567, 14.4888306
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3686333, 9.3659172
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8162766, 7.8149643
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6346283, 7.6303902
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7057381, 8.7042007
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7498131, 7.7459869
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5055504, 8.5018616
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5527725, 8.5428963
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3329926, 13.3269043
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3946800, 10.3926620
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6067505, 8.6012001
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3701515, 8.3617058
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3877831, 10.3859177
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2725220, 10.2826538
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6222572, 12.6431580
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2579536, 10.2683372
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4033203, 10.4154015
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0114136, 13.0228157
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2810402, 12.2998924
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9800797, 11.9873466
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0796814, 13.1044044
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0874634, 9.1037636
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5467606, 11.5600739
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7761459, 7.7796326

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6247962, upper bound: 5.6176692
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6258148, upper bound: 5.6166855
time: 5.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1041565, 16.1039658
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4136124, 8.4118690
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7158966, 9.7165184
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9389267, 8.9353981
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8115463, 10.8105507
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6138763, 11.6142273
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3279839, 10.3363419
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0184402, 11.0186157
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.5994415, 12.5959702
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1135178, 9.1032372
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3616180, 12.3489189
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8434448, 6.8364220
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2741776, 13.2750549
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5347366, 13.5396156
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7374115, 17.7294846
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1978378, 9.1956596
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4445381, 7.4398365
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4842377, 14.4930496
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3663864, 9.3681641
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8161850, 7.8150558
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6337662, 7.6312542
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7053299, 8.7046089
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7499504, 7.7458458
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5036850, 8.5037231
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5514946, 8.5441742
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3315201, 13.3283691
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3947258, 10.3926201
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6049500, 8.6030045
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3681450, 8.3637161
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3872261, 10.3864746
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2733612, 10.2818146
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6244698, 12.6409454
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2585220, 10.2677650
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4059677, 10.4127579
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0107574, 13.0234642
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2822151, 12.2987137
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9770050, 11.9904137
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0814362, 13.1026573
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0861473, 9.1050797
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5475769, 11.5592537
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7760849, 7.7796898

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6186611, upper bound: 5.6238441
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6196356, upper bound: 5.6228169
time: 13.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15.2007008, 1.7268648, -15.2007008, 1.7268648, -16.1042938, 16.1038437
1: -4.0436954, 5.7076054, -4.0436954, 5.7076054, -8.4135208, 8.4119568
2: -5.3023701, 6.1819921, -5.3023701, 6.1819921, -9.7160110, 9.7164040
3: -13.6200066, -1.6134195, -13.6200066, -1.6134195, -8.9375534, 8.9367714
4: -7.5437779, 6.2721972, -7.5437779, 6.2721972, -10.8115692, 10.8105278
5: -9.7356529, 3.9496644, -9.7356529, 3.9496644, -11.6138306, 11.6142654
6: -25.1056499, -11.1789246, -25.1056499, -11.1789246, -10.3262291, 10.3380966
7: -9.8104610, 4.9173517, -9.8104610, 4.9173517, -11.0179443, 11.0190964
8: -15.5054245, 1.4822176, -15.5054245, 1.4822176, -12.6002502, 12.5951538
9: -11.0629692, 1.1509449, -11.0629692, 1.1509449, -9.1120834, 9.1046677
10: -6.3927660, 8.3376350, -6.3927660, 8.3376350, -12.3614426, 12.3490906
11: -0.7499009, 9.0063362, -0.7499009, 9.0063362, -6.8434448, 6.8364220
12: -17.7487297, 0.5295393, -17.7487297, 0.5295393, -13.2738190, 13.2753601
13: -25.8000565, -7.4265175, -25.8000565, -7.4265175, -13.5339661, 13.5403786
14: -23.2131920, -0.5969784, -23.2131920, -0.5969784, -17.7399216, 17.7269745
15: -10.8789263, 0.2424475, -10.8789263, 0.2424475, -9.1996574, 9.1938400
16: -9.3277349, 1.6277586, -9.3277349, 1.6277586, -7.4437370, 7.4406376
17: -14.4211092, 3.9907079, -14.4211092, 3.9907079, -14.4872284, 14.4900665
18: 4.2878771, 15.7111120, 4.2878771, 15.7111120, -9.3681564, 9.3663979
19: 0.8949819, 8.9344292, 0.8949819, 8.9344292, -7.8166504, 7.8145828
20: -0.3983452, 7.7920113, -0.3983452, 7.7920113, -7.6334114, 7.6315994
21: -0.0313234, 9.1579657, -0.0313234, 9.1579657, -9.1892891, 9.1892891
22: -0.7954416, 9.9751968, -0.7954416, 9.9751968, -8.7061577, 8.7037811
23: 0.0487075, 10.0184164, 0.0487075, 10.0184164, -7.7508392, 7.7449303
24: 3.2032988, 14.3682671, 3.2032988, 14.3682671, -8.5054741, 8.5019341
25: 2.1996024, 13.4027576, 2.1996024, 13.4027576, -8.5532227, 8.5424461
26: -4.8805842, 10.9047451, -4.8805842, 10.9047451, -13.3334198, 13.3264771
27: -3.1976068, 9.9512768, -3.1976068, 9.9512768, -13.1488838, 13.1488838
28: 0.9346018, 12.3529654, 0.9346018, 12.3529654, -10.3950310, 10.3923111
29: -2.3487060, 9.5128765, -2.3487060, 9.5128765, -8.6063042, 8.6016464
30: -1.6380250, 9.5926991, -1.6380250, 9.5926991, -8.3681030, 8.3637581
31: 1.8034883, 12.2705917, 1.8034883, 12.2705917, -10.3880348, 10.3856621
32: -22.6536732, -9.2068043, -22.6536732, -9.2068043, -10.2707024, 10.2844734
33: -36.3427734, -16.0448608, -36.3427734, -16.0448608, -12.6220207, 12.6433983
34: -28.7373981, -13.0149841, -28.7373981, -13.0149841, -10.2564011, 10.2698860
35: -23.2304192, -6.8863297, -23.2304192, -6.8863297, -10.4041595, 10.4145622
36: -26.0963993, -5.8661323, -26.0963993, -5.8661323, -13.0103149, 13.0238914
37: -36.8791466, -16.4219360, -36.8791466, -16.4219360, -12.2827034, 12.2982254
38: -26.1904259, -6.1886387, -26.1904259, -6.1886387, -11.9765701, 11.9908104
39: -40.5990067, -18.9726067, -40.5990067, -18.9726067, -13.0798645, 13.1042290
40: -36.8379402, -22.6805077, -36.8379402, -22.6805077, -9.0862160, 9.1050110
41: -24.1266880, -8.2118998, -24.1266880, -8.2118998, -11.5460587, 11.5607758
42: -21.3246174, -11.0693035, -21.3246174, -11.0693035, -7.7752762, 7.7804718

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 1503
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1326
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1519
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1289
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1737

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6188300, upper bound: 5.6236952
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.6197815, upper bound: 5.6226265
time: 15.03 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 23.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6080444, upper bound: 5.6344492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6090274, upper bound: 5.6334336
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6082126, upper bound: 5.6342939
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6091703, upper bound: 5.6332386
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6248907, upper bound: 5.6176099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6258632, upper bound: 5.6165805
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6250599, upper bound: 5.6174615
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6260097, upper bound: 5.6163903
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6077501, upper bound: 5.6346429
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6088196, upper bound: 5.6336959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6079385, upper bound: 5.6344984
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6089675, upper bound: 5.6335263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6246025, upper bound: 5.6178122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6256610, upper bound: 5.6168541
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6247962, upper bound: 5.6176692
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6258148, upper bound: 5.6166855
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6186611, upper bound: 5.6238441
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6196356, upper bound: 5.6228169
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6188300, upper bound: 5.6236952
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 23.40
Output dim: 18, lower bound: -5.6197815, upper bound: 5.6226265
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6385655, upper bound: 5.6090754
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6387390, upper bound: 5.6089018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6215604, upper bound: 5.6261314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6217406, upper bound: 5.6259606
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6384161, upper bound: 5.6092312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6386036, upper bound: 5.6090638
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6108374, upper bound: 5.6368424
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6110043, upper bound: 5.6366584
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6277357, upper bound: 5.6199639
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6279071, upper bound: 5.6197835
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6106739, upper bound: 5.6369862
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6108468, upper bound: 5.6368132
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6275799, upper bound: 5.6201117
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6277602, upper bound: 5.6199415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6214966, upper bound: 5.6262053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6216668, upper bound: 5.6260246
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6383523, upper bound: 5.6093020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6385256, upper bound: 5.6091270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6213389, upper bound: 5.6263531
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6215185, upper bound: 5.6261822
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6382026, upper bound: 5.6094553
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6383890, upper bound: 5.6092880
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6092880, upper bound: 5.6383890
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6094553, upper bound: 5.6382026
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6261822, upper bound: 5.6215185
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6263531, upper bound: 5.6213389
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6091270, upper bound: 5.6385256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6093020, upper bound: 5.6383523
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6260246, upper bound: 5.6216668
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6262053, upper bound: 5.6214966
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6199415, upper bound: 5.6277602
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6201117, upper bound: 5.6275799
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6368132, upper bound: 5.6108468
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6369862, upper bound: 5.6106739
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6197835, upper bound: 5.6279071
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6199639, upper bound: 5.6277357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6366584, upper bound: 5.6110043
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6368424, upper bound: 5.6108374
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6090638, upper bound: 5.6386036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6092312, upper bound: 5.6384161
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6259606, upper bound: 5.6217406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6261314, upper bound: 5.6215604
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6089018, upper bound: 5.6387390
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6090754, upper bound: 5.6385655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6258030, upper bound: 5.6218874
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6259834, upper bound: 5.6217166
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6197198, upper bound: 5.6279813
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6198900, upper bound: 5.6277992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6365948, upper bound: 5.6110748
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6367674, upper bound: 5.6109011
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6195617, upper bound: 5.6281257
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6197414, upper bound: 5.6279540
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6364390, upper bound: 5.6112312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.40
Output dim: 18, lower bound: -5.6366220, upper bound: 5.6110639

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 27.15 + 1790.05 = 1817.20 seconds
