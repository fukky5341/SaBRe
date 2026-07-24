## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 37.4874905906


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=153, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.1157036, 35.2542267, -43.1157036, 35.2542267, -78.3699341, 78.3699341)
1: (-23.3525658, 32.0399094, -23.3525658, 32.0399094, -55.3924751, 55.3924751)
2: (-18.8440971, 31.9421139, -18.8440971, 31.9421139, -50.7670898, 50.7670937)
3: (-19.0323219, 35.1756439, -19.0323219, 35.1756439, -53.7754364, 53.7754440)
4: (-23.5116367, 36.0320663, -23.5116367, 36.0320663, -59.5437012, 59.5437012)
5: (-21.1989937, 35.5179977, -21.1989937, 35.5179977, -56.0920868, 56.0921021)
6: (-42.1937408, 26.1332111, -42.1937408, 26.1332111, -68.3269501, 68.3269501)
7: (-30.4197502, 34.2613983, -30.4197502, 34.2613983, -64.0413132, 64.0413132)
8: (-29.0230789, 40.1350403, -29.0230789, 40.1350403, -69.1581192, 69.1581192)
9: (-24.4200497, 31.6706371, -24.4200497, 31.6706371, -54.9636116, 54.9636154)
10: (-45.8945236, 31.3143044, -45.8945236, 31.3143044, -77.2088318, 77.2088318)
11: (-48.9226913, 18.2249794, -48.9226913, 18.2249794, -67.1476746, 67.1476746)
12: (-52.8700562, 18.2259483, -52.8700562, 18.2259483, -69.4354858, 69.4354935)
13: (-35.8046913, 38.7107353, -35.8046913, 38.7107353, -74.5154266, 74.5154266)
14: (-78.4954071, 11.0890770, -78.4954071, 11.0890770, -89.5844879, 89.5844879)
15: (-30.4171829, 30.1431236, -30.4171829, 30.1431236, -60.5603065, 60.5603065)
16: (-46.2954254, 30.9688759, -46.2954254, 30.9688759, -77.2572174, 77.2572327)
17: (-77.9650040, 14.7196960, -77.9650040, 14.7196960, -92.6847000, 92.6847000)
18: (-45.8355713, 21.3369789, -45.8355713, 21.3369789, -67.1725464, 67.1725464)
19: (-34.5261383, 11.0005360, -34.5261383, 11.0005360, -45.5266724, 45.5266724)
20: (-30.6183167, 14.3167229, -30.6183167, 14.3167229, -44.9350395, 44.9350395)
21: (-42.7198906, 14.9663868, -42.7198906, 14.9663868, -57.6862793, 57.6862793)
22: (-43.3567314, 17.6444550, -43.3567314, 17.6444550, -61.0011864, 61.0011864)
23: (-34.4368858, 15.1919584, -34.4368858, 15.1919584, -49.6288452, 49.6288452)
24: (-36.4291306, 14.9190760, -36.4291306, 14.9190760, -51.3482056, 51.3482056)
25: (-35.6012077, 17.3636513, -35.6012077, 17.3636513, -52.9648590, 52.9648590)
26: (-53.6148529, 20.2797928, -53.6148529, 20.2797928, -73.8946457, 73.8946457)
27: (-36.2488594, 18.9731464, -36.2488594, 18.9731464, -55.2220078, 55.2220078)
28: (-33.3455009, 19.0365372, -33.3455009, 19.0365372, -52.3820381, 52.3820381)
29: (-45.0150070, 16.8973007, -45.0150070, 16.8973007, -61.9123077, 61.9123077)
30: (-42.8835983, 20.0797844, -42.8835983, 20.0797844, -62.9633827, 62.9633827)
31: (-42.3394241, 15.3661118, -42.3394241, 15.3661118, -57.7055359, 57.7055359)
32: (-38.5433426, 23.2086086, -38.5433426, 23.2086086, -61.7519531, 61.7519531)
33: (-48.8919144, 35.9871902, -48.8919144, 35.9871902, -84.8791046, 84.8791046)
34: (-47.1912689, 21.1381664, -47.1912689, 21.1381664, -68.2930145, 68.2930222)
35: (-41.7281647, 26.4423904, -41.7281647, 26.4423904, -67.7198257, 67.7198334)
36: (-42.4851074, 26.6478462, -42.4851074, 26.6478462, -68.2822266, 68.2822189)
37: (-66.9026871, 22.3323860, -66.9026871, 22.3323860, -86.7244415, 86.7244415)
38: (-52.5966721, 31.2848778, -52.5966721, 31.2848778, -82.0830994, 82.0831070)
39: (-60.3237343, 35.4661560, -60.3237343, 35.4661560, -95.7898865, 95.7898865)
40: (-53.5796928, 28.4329319, -53.5796928, 28.4329319, -82.0126266, 82.0126266)
41: (-39.1193237, 27.1914368, -39.1193237, 27.1914368, -66.3107605, 66.3107605)
42: (-32.5631599, 22.0211430, -32.5631599, 22.0211430, -54.5843048, 54.5843048)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.80 + 78.09 = 80.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -37.5438063, upper bound: 37.5438063

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 664

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5368227, upper bound: 37.4945694
time: 52.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5368227, upper bound: 37.5368226
time: 41.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 94.31 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 94.31
Output dim: 8, lower bound: -37.5368227, upper bound: 37.4945694
IS_A2, status: Status.UNKNOWN, split count: 1, time: 94.31
Output dim: 8, lower bound: -37.5368227, upper bound: 37.5368226

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -43.0954170, 35.2237129, -43.1060905, 35.2398491, -78.3352661, 78.3298035
1: -23.3429756, 32.0092087, -23.3480167, 32.0253448, -55.3683205, 55.3572235
2: -18.8342476, 31.8894882, -18.8394012, 31.9167786, -50.7317657, 50.7095375
3: -19.0215969, 35.1164627, -19.0272655, 35.1471100, -53.7368469, 53.7121735
4: -23.5029926, 36.0142479, -23.5074921, 36.0235405, -59.5265350, 59.5217400
5: -21.1891327, 35.4563828, -21.1942844, 35.4885330, -56.0526886, 56.0253258
6: -42.1778297, 26.0877419, -42.1861191, 26.1115799, -68.2894135, 68.2738647
7: -30.4044991, 34.1690025, -30.4124775, 34.2170525, -63.9815521, 63.9414597
8: -29.0148277, 40.0810280, -29.0191841, 40.1095428, -69.1243744, 69.1002121
9: -24.4039230, 31.6560040, -24.4122963, 31.6636925, -54.9290085, 54.9354172
10: -45.8537750, 31.2917137, -45.8751907, 31.3036041, -77.1573792, 77.1669006
11: -48.8903236, 18.1710320, -48.9072723, 18.1988182, -67.0891418, 67.0783081
12: -52.7025566, 18.2073860, -52.7910461, 18.2171955, -69.2573242, 69.3358154
13: -35.7508011, 38.6898346, -35.7770424, 38.7008286, -74.4516296, 74.4668732
14: -78.3778076, 11.0785370, -78.4393082, 11.0839863, -89.4617920, 89.5178452
15: -30.3152962, 30.1257286, -30.3689747, 30.1347122, -60.4500084, 60.4947052
16: -46.2670822, 30.8863449, -46.2817764, 30.9296837, -77.1895599, 77.1613464
17: -77.8407364, 14.7045555, -77.9049988, 14.7123489, -92.5530853, 92.6095581
18: -45.7780609, 21.3241692, -45.8080177, 21.3306808, -67.1087418, 67.1321869
19: -34.5044861, 10.9943466, -34.5157700, 10.9975929, -45.5020790, 45.5101166
20: -30.5878525, 14.3087893, -30.6038380, 14.3129807, -44.9008331, 44.9126282
21: -42.6958389, 14.9548264, -42.7083893, 14.9608507, -57.6566887, 57.6632156
22: -43.2440262, 17.6309662, -43.3028984, 17.6379299, -60.8819580, 60.9338646
23: -34.4127998, 15.1739445, -34.4254456, 15.1830873, -49.5958862, 49.5993881
24: -36.4095917, 14.8923492, -36.4199104, 14.9057541, -51.3153458, 51.3122597
25: -35.5681534, 17.3526535, -35.5855446, 17.3583412, -52.9264946, 52.9381981
26: -53.4450684, 20.2633362, -53.5351257, 20.2717533, -73.7168198, 73.7984619
27: -36.2264442, 18.9589539, -36.2382622, 18.9663658, -55.1928101, 55.1972160
28: -33.3251114, 19.0264397, -33.3358154, 19.0316219, -52.3567352, 52.3622551
29: -44.9303131, 16.8901520, -44.9745255, 16.8939171, -61.8242302, 61.8646774
30: -42.8605499, 19.9933949, -42.8726044, 20.0383968, -62.8989487, 62.8659973
31: -42.3180542, 15.3526907, -42.3291702, 15.3597584, -57.6778107, 57.6818619
32: -38.4867363, 23.1926155, -38.5167007, 23.2010651, -61.6878014, 61.7093163
33: -48.8720512, 35.9489861, -48.8825378, 35.9691544, -84.8412018, 84.8315277
34: -47.1722984, 21.1174507, -47.1823120, 21.1282349, -68.2641907, 68.2633362
35: -41.7093468, 26.4273186, -41.7192535, 26.4351959, -67.6912003, 67.6903763
36: -42.4261322, 26.6389198, -42.4568176, 26.6436195, -68.2222748, 68.2463913
37: -66.8645706, 22.3170242, -66.8845673, 22.3251820, -86.6807251, 86.6896286
38: -52.5192871, 31.2664585, -52.5573158, 31.2760735, -82.0035706, 82.0303497
39: -60.3027573, 35.4433594, -60.3136063, 35.4554825, -95.7582397, 95.7569656
40: -53.5622559, 28.3681107, -53.5713959, 28.4017315, -81.9639893, 81.9395065
41: -39.1010513, 27.1677628, -39.1105881, 27.1798229, -66.2808762, 66.2783508
42: -32.5444832, 21.9951515, -32.5542831, 22.0086384, -54.5531235, 54.5494347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
time: 48.39 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
time: 192.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -43.2148323, 35.2565880, -43.1122818, 35.2408600, -78.4556885, 78.3688660
1: -23.4339752, 32.0395660, -23.3504810, 32.0254211, -55.4593964, 55.3900452
2: -18.9520550, 31.9533195, -18.8422928, 31.9366760, -50.8698425, 50.7758904
3: -19.1533585, 35.1931381, -19.0305233, 35.1675110, -53.8892822, 53.7872772
4: -23.5593987, 36.0480080, -23.5098171, 36.0279922, -59.5873909, 59.5578232
5: -21.3257942, 35.5388260, -21.1973705, 35.5117340, -56.2154770, 56.1089401
6: -42.2311478, 26.1493454, -42.1897278, 26.1168003, -68.3479462, 68.3390732
7: -30.6046848, 34.2577019, -30.4166565, 34.2391853, -64.2035217, 64.0346451
8: -29.1349773, 40.1578903, -29.0206013, 40.1294556, -69.2644348, 69.1784897
9: -24.4481716, 31.7155628, -24.4143906, 31.6681633, -54.9782028, 55.0336342
10: -45.9322586, 31.4228058, -45.8889542, 31.3116627, -77.2439194, 77.3117599
11: -48.9974365, 18.2509689, -48.9159050, 18.1946907, -67.1921234, 67.1668701
12: -52.8987999, 18.5276661, -52.8629990, 18.2237263, -69.4535446, 69.7327499
13: -35.8345146, 38.8438950, -35.7896729, 38.7070541, -74.5415649, 74.6335678
14: -78.5447617, 11.2574730, -78.4861832, 11.0870094, -89.6317749, 89.7436523
15: -30.4365196, 30.2884560, -30.3855896, 30.1387348, -60.5752563, 60.6740456
16: -46.4256363, 30.9579620, -46.2887650, 30.9320545, -77.3480988, 77.2371216
17: -77.9895935, 14.9316120, -77.9566574, 14.7148075, -92.7043991, 92.8882675
18: -45.8623276, 21.4116936, -45.8267975, 21.3327103, -67.1950378, 67.2384949
19: -34.5395584, 11.0167484, -34.5224838, 10.9900875, -45.5296478, 45.5392303
20: -30.6564350, 14.3634853, -30.6146488, 14.3149672, -44.9714012, 44.9781342
21: -42.7519760, 14.9857302, -42.7158737, 14.9490709, -57.7010460, 57.7016029
22: -43.3746910, 17.8336773, -43.3340454, 17.6402473, -61.0149384, 61.1677246
23: -34.4764481, 15.2044506, -34.4339790, 15.1846743, -49.6611214, 49.6384277
24: -36.5104370, 14.9376049, -36.4258957, 14.9148655, -51.4253006, 51.3635025
25: -35.6146507, 17.4266033, -35.5877686, 17.3610268, -52.9756775, 53.0143738
26: -53.6512413, 20.5485306, -53.5969887, 20.2750626, -73.9263000, 74.1455231
27: -36.3227844, 18.9745502, -36.2456055, 18.9597206, -55.2825050, 55.2201538
28: -33.3886223, 19.0598221, -33.3430481, 19.0325966, -52.4212189, 52.4028702
29: -45.0508766, 17.0683060, -45.0078506, 16.8953781, -61.9462547, 62.0761566
30: -43.0100021, 20.1089935, -42.8782921, 20.0710869, -63.0810890, 62.9872856
31: -42.3935242, 15.3824005, -42.3353043, 15.3468838, -57.7404099, 57.7177048
32: -38.5757599, 23.2928638, -38.5389977, 23.2057076, -61.7814674, 61.8318634
33: -48.9713364, 36.0255089, -48.8886185, 35.9824066, -84.9537430, 84.9141235
34: -47.2310715, 21.1632385, -47.1875534, 21.1342049, -68.3309937, 68.3143616
35: -41.7647972, 26.4622574, -41.7250938, 26.4394550, -67.7643967, 67.7279205
36: -42.5013809, 26.7935963, -42.4666443, 26.6463184, -68.2995758, 68.4054031
37: -66.9332886, 22.3824692, -66.8901062, 22.3299561, -86.7695770, 86.7622299
38: -52.6494751, 31.4284573, -52.5875473, 31.2811089, -82.1324005, 82.2139969
39: -60.3455429, 35.5008316, -60.3142548, 35.4633789, -95.8089218, 95.8150864
40: -53.6747589, 28.4401398, -53.5756950, 28.4213352, -82.0960922, 82.0158386
41: -39.1556854, 27.2170506, -39.1157455, 27.1832123, -66.3388977, 66.3327942
42: -32.5823059, 22.0656700, -32.5586700, 22.0139732, -54.5962791, 54.6243401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=153, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836
time: 46.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836
time: 56.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 105.26 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 105.26
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 105.26
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 105.26
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 105.26
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -43.1873245, 35.2286453, -42.9832230, 35.1472626, -78.3345871, 78.2118683
1: -23.4206295, 32.0196953, -23.2788982, 31.9494324, -55.3700638, 55.2985916
2: -18.9362297, 31.9059753, -18.7296257, 31.8219547, -50.7385101, 50.6146469
3: -19.1384468, 35.1201744, -18.8820782, 34.9882355, -53.6911774, 53.5541039
4: -23.5326862, 36.0338593, -23.4186554, 35.9655418, -59.4982300, 59.4525146
5: -21.3093815, 35.4631348, -21.0318604, 35.3196754, -56.0068512, 55.8613052
6: -42.2185020, 26.0871410, -42.1171417, 25.9540653, -68.1725693, 68.2042847
7: -30.5832596, 34.1818085, -30.2421818, 34.0481339, -63.9878998, 63.7758026
8: -29.1223068, 40.1205215, -28.9149055, 40.0170670, -69.1393738, 69.0354309
9: -24.3838043, 31.7046185, -24.2503510, 31.5313244, -54.7523727, 54.8556252
10: -45.8144226, 31.4022446, -45.6131439, 31.0689278, -76.8833466, 77.0153885
11: -48.9693642, 18.2227440, -48.8397293, 18.0595360, -67.0289001, 67.0624695
12: -52.7287521, 18.5054855, -52.4739914, 17.8985004, -68.9462128, 69.3125153
13: -35.7783165, 38.8261948, -35.6193771, 38.5770950, -74.3554077, 74.4455719
14: -78.4024734, 11.2494392, -78.1335449, 10.8737984, -89.2762756, 89.3829803
15: -30.3447075, 30.2741661, -30.1439781, 29.9719086, -60.3166161, 60.4181442
16: -46.3919868, 30.9206524, -46.2257614, 30.7980633, -77.1809387, 77.1366882
17: -77.8752441, 14.9162922, -77.6970367, 14.4757042, -92.3509521, 92.6133270
18: -45.8158493, 21.3954220, -45.6944313, 21.2265091, -67.0423584, 67.0898514
19: -34.5169945, 11.0100203, -34.4516449, 10.9509821, -45.4679756, 45.4616661
20: -30.6291714, 14.3542175, -30.5174828, 14.2630863, -44.8922577, 44.8717003
21: -42.7272339, 14.9750681, -42.6402054, 14.8819847, -57.6092186, 57.6152725
22: -43.2729111, 17.8209229, -43.0845070, 17.4612942, -60.7342072, 60.9054298
23: -34.4553223, 15.1782713, -34.3491211, 15.1072178, -49.5625381, 49.5273933
24: -36.4931641, 14.9148722, -36.3365440, 14.8474846, -51.3406487, 51.2514153
25: -35.5863419, 17.4120350, -35.4971619, 17.2844830, -52.8708267, 52.9091949
26: -53.4974518, 20.5275631, -53.2306480, 20.0038033, -73.5012512, 73.7582092
27: -36.3065414, 18.9448204, -36.1279144, 18.8759308, -55.1824722, 55.0727348
28: -33.3722420, 19.0263920, -33.2466888, 18.9393597, -52.3115997, 52.2730789
29: -44.9864082, 17.0576439, -44.8299561, 16.7251663, -61.7115746, 61.8875999
30: -42.9896889, 20.0381050, -42.7162552, 19.8807182, -62.8704071, 62.7543602
31: -42.3721924, 15.3691530, -42.2878761, 15.2820559, -57.6542473, 57.6570282
32: -38.5381012, 23.2747459, -38.4174957, 23.1167564, -61.6548576, 61.6922417
33: -48.9460373, 35.9984741, -48.7372971, 35.8883820, -84.8344193, 84.7357712
34: -47.2152634, 21.1077194, -47.0639458, 20.9894028, -68.1701202, 68.1332397
35: -41.7459755, 26.4087296, -41.5867233, 26.3085957, -67.6146851, 67.5253754
36: -42.4826393, 26.7775707, -42.3758545, 26.6041622, -68.2413254, 68.2960205
37: -66.8963013, 22.3654175, -66.7516174, 22.2568512, -86.6518860, 86.5931091
38: -52.6346130, 31.3970566, -52.4874649, 31.1875229, -82.0225677, 82.0658340
39: -60.3002090, 35.4789352, -60.1567268, 35.3691254, -95.6693344, 95.6356659
40: -53.6524544, 28.3949699, -53.4272156, 28.3136635, -81.9661179, 81.8221893
41: -39.1421509, 27.1788330, -39.0479698, 27.0733337, -66.2154846, 66.2268066
42: -32.5695152, 22.0322952, -32.5050964, 21.9015598, -54.4710770, 54.5373917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
time: 67.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.5234558
time: 53.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -43.2108459, 35.2454758, -43.1042862, 35.2176971, -78.4285431, 78.3497620
1: -23.4316597, 32.0334778, -23.3458462, 32.0130806, -55.4447403, 55.3793259
2: -18.9498100, 31.9506760, -18.8377838, 31.9311752, -50.8599777, 50.7686501
3: -19.1509705, 35.1829262, -19.0257435, 35.1471825, -53.8516159, 53.7715797
4: -23.5530357, 36.0459061, -23.4963913, 36.0236740, -59.5767097, 59.5422974
5: -21.3233948, 35.5278130, -21.1925259, 35.4929962, -56.1882401, 56.0937386
6: -42.2287331, 26.1325455, -42.1848793, 26.0812702, -68.3100052, 68.3174286
7: -30.6015396, 34.2439346, -30.4103298, 34.2128563, -64.1650467, 64.0139542
8: -29.1318455, 40.1550598, -29.0142536, 40.1240845, -69.2559280, 69.1693115
9: -24.4442291, 31.7133789, -24.4064865, 31.6636868, -54.9865036, 55.0168648
10: -45.9270859, 31.4201946, -45.8780594, 31.3064308, -77.2335205, 77.2982559
11: -48.9933167, 18.2267933, -48.9076195, 18.1428852, -67.1362000, 67.1344147
12: -52.8921700, 18.5249691, -52.8492813, 18.2183323, -69.4407959, 69.6890564
13: -35.8254929, 38.8407364, -35.7740631, 38.7005539, -74.5260468, 74.6147995
14: -78.5371246, 11.2560368, -78.4710541, 11.0839100, -89.6210327, 89.7270889
15: -30.4096489, 30.2860394, -30.3381290, 30.1338673, -60.5435181, 60.6241684
16: -46.4220009, 30.9316025, -46.2816162, 30.8769093, -77.2897873, 77.2036438
17: -77.9798126, 14.9286461, -77.9365082, 14.7087936, -92.6886063, 92.8651581
18: -45.8538284, 21.4092827, -45.8085785, 21.3278027, -67.1816330, 67.2178650
19: -34.5371361, 11.0148306, -34.5175705, 10.9854774, -45.5226135, 45.5324020
20: -30.6537819, 14.3616390, -30.6094055, 14.3112373, -44.9650192, 44.9710464
21: -42.7491417, 14.9806957, -42.7101288, 14.9374790, -57.6866226, 57.6908264
22: -43.3582535, 17.8312340, -43.3056374, 17.6353588, -60.9936142, 61.1368713
23: -34.4744568, 15.1971893, -34.4300270, 15.1695223, -49.6439781, 49.6272163
24: -36.5075378, 14.9325962, -36.4199371, 14.9051418, -51.4126816, 51.3525314
25: -35.6064835, 17.4242001, -35.5705299, 17.3562813, -52.9627647, 52.9947281
26: -53.6387672, 20.5459938, -53.5712128, 20.2700195, -73.9087830, 74.1172028
27: -36.3203125, 18.9640884, -36.2405167, 18.9407921, -55.2611046, 55.2046051
28: -33.3867416, 19.0557289, -33.3393250, 19.0244675, -52.4112091, 52.3950539
29: -45.0467529, 17.0662575, -44.9991417, 16.8913422, -61.9380951, 62.0653992
30: -43.0062866, 20.1041298, -42.8707962, 20.0612316, -63.0675201, 62.9749260
31: -42.3908920, 15.3729115, -42.3299789, 15.3263674, -57.7172585, 57.7028885
32: -38.5726547, 23.2899284, -38.5327835, 23.1996822, -61.7723389, 61.8227119
33: -48.9673157, 36.0223236, -48.8806572, 35.9760971, -84.9434128, 84.9029846
34: -47.2288589, 21.1599407, -47.1830521, 21.1275616, -68.3216248, 68.3076324
35: -41.7617836, 26.4589500, -41.7190361, 26.4327126, -67.7516022, 67.7258530
36: -42.4978256, 26.7919254, -42.4574203, 26.6428757, -68.2909698, 68.4014816
37: -66.9237747, 22.3802319, -66.8705139, 22.3254700, -86.7536697, 86.7352753
38: -52.6467438, 31.4251080, -52.5810318, 31.2742748, -82.1177063, 82.2046051
39: -60.3350639, 35.4983253, -60.2926865, 35.4585114, -95.7935791, 95.7910156
40: -53.6715622, 28.4353981, -53.5694656, 28.4124203, -82.0839844, 82.0048676
41: -39.1535835, 27.2079048, -39.1115112, 27.1639481, -66.3175354, 66.3194122
42: -32.5801468, 22.0588303, -32.5542793, 22.0001564, -54.5803032, 54.6131096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=152, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
time: 42.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5234559, upper bound: 37.5234558
time: 53.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 98.41 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 98.41
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 98.41
Output dim: 8, lower bound: -37.4795223, upper bound: 37.5234558
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 98.41
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 98.41
Output dim: 8, lower bound: -37.5234559, upper bound: 37.5234558

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.1791306, 35.2180061, -42.9794960, 35.1423340, -78.3214645, 78.1975021
1: -23.4156837, 32.0130234, -23.2765884, 31.9456711, -55.3613548, 55.2896118
2: -18.9316292, 31.9016724, -18.7275124, 31.8199806, -50.7318726, 50.6080818
3: -19.1341248, 35.1137695, -18.8800869, 34.9851723, -53.6835327, 53.5443192
4: -23.5259895, 36.0299835, -23.4152470, 35.9637146, -59.4897041, 59.4452286
5: -21.3052120, 35.4561691, -21.0299397, 35.3164177, -55.9992981, 55.8517456
6: -42.2137909, 26.0650806, -42.1149139, 25.9438114, -68.1576004, 68.1799927
7: -30.5784588, 34.1746979, -30.2399406, 34.0438271, -63.9792480, 63.7653427
8: -29.1172810, 40.1156960, -28.9126244, 40.0148201, -69.1321030, 69.0283203
9: -24.3747368, 31.7009621, -24.2461281, 31.5296612, -54.7420578, 54.8488579
10: -45.8070145, 31.3946953, -45.6096497, 31.0654202, -76.8724365, 77.0043488
11: -48.9620895, 18.1815739, -48.8362961, 18.0385418, -67.0006332, 67.0178680
12: -52.7196350, 18.5016365, -52.4696960, 17.8967476, -68.9330139, 69.3037567
13: -35.7549820, 38.8215790, -35.6084900, 38.5749054, -74.3298874, 74.4300690
14: -78.3921661, 11.2476559, -78.1287003, 10.8729420, -89.2651062, 89.3763580
15: -30.3128643, 30.2693329, -30.1291027, 29.9696026, -60.2824669, 60.3984375
16: -46.3828430, 30.8877411, -46.2214508, 30.7827187, -77.1565094, 77.0994797
17: -77.8674011, 14.9097748, -77.6920013, 14.4726658, -92.3400650, 92.6017761
18: -45.8119469, 21.3903885, -45.6926117, 21.2241135, -67.0360565, 67.0830002
19: -34.5133667, 11.0021095, -34.4499245, 10.9471560, -45.4605217, 45.4520340
20: -30.6252251, 14.3502998, -30.5155926, 14.2609234, -44.8861465, 44.8658905
21: -42.7224045, 14.9589119, -42.6379166, 14.8742352, -57.5966415, 57.5968285
22: -43.2532959, 17.8152466, -43.0729218, 17.4586258, -60.7119217, 60.8881683
23: -34.4520531, 15.1657267, -34.3475761, 15.1013899, -49.5534439, 49.5133018
24: -36.4865913, 14.9054146, -36.3334656, 14.8430557, -51.3296471, 51.2388802
25: -35.5785332, 17.4075451, -35.4931107, 17.2823906, -52.8609238, 52.9006577
26: -53.4881287, 20.5217991, -53.2262383, 20.0010700, -73.4891968, 73.7480392
27: -36.3004074, 18.9310608, -36.1250648, 18.8687649, -55.1691742, 55.0561256
28: -33.3694839, 19.0199890, -33.2453728, 18.9362087, -52.3056946, 52.2653618
29: -44.9738312, 17.0530777, -44.8236122, 16.7230797, -61.6969109, 61.8766899
30: -42.9826126, 20.0147209, -42.7129211, 19.8697624, -62.8523750, 62.7276421
31: -42.3685608, 15.3525200, -42.2861252, 15.2742577, -57.6428185, 57.6386452
32: -38.5303955, 23.2708168, -38.4135284, 23.1149311, -61.6453247, 61.6843452
33: -48.9409142, 35.9943619, -48.7348328, 35.8864899, -84.8274078, 84.7291946
34: -47.2124939, 21.1009541, -47.0626526, 20.9860573, -68.1640778, 68.1251373
35: -41.7419815, 26.4050159, -41.5847702, 26.3068390, -67.6093140, 67.5194931
36: -42.4662209, 26.7757397, -42.3680687, 26.6032639, -68.2223663, 68.2851791
37: -66.8841705, 22.3617668, -66.7458878, 22.2551346, -86.6376801, 86.5830154
38: -52.6301727, 31.3916264, -52.4855652, 31.1847382, -82.0151596, 82.0577698
39: -60.2766953, 35.4756050, -60.1456642, 35.3675346, -95.6442261, 95.6212692
40: -53.6444969, 28.3833179, -53.4234734, 28.3081303, -81.9526291, 81.8067932
41: -39.1375999, 27.1676216, -39.0458260, 27.0677662, -66.2053680, 66.2134476
42: -32.5656662, 22.0182781, -32.5032730, 21.8949738, -54.4606400, 54.5215530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 48.71 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 58.22 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.2027740, 35.2357521, -43.1006279, 35.2129822, -78.4157562, 78.3363800
1: -23.4268188, 32.0272179, -23.3436604, 32.0101166, -55.4369354, 55.3708801
2: -18.9452839, 31.9464779, -18.8357487, 31.9292603, -50.8535004, 50.7622414
3: -19.1467381, 35.1766739, -19.0238419, 35.1443253, -53.8442383, 53.7620163
4: -23.5465069, 36.0420418, -23.4933357, 36.0219193, -59.5684280, 59.5353775
5: -21.3193016, 35.5209808, -21.1907005, 35.4899521, -56.1809540, 56.0844383
6: -42.2240906, 26.1105747, -42.1827927, 26.0713177, -68.2954102, 68.2933655
7: -30.5967712, 34.2368393, -30.4081593, 34.2096329, -64.1576385, 64.0051575
8: -29.1268578, 40.1502762, -29.0120258, 40.1218643, -69.2487183, 69.1623001
9: -24.4353657, 31.7097816, -24.4024334, 31.6621017, -54.9765778, 55.0102615
10: -45.9196625, 31.4127579, -45.8747025, 31.3031120, -77.2227783, 77.2874603
11: -48.9861679, 18.1874428, -48.9043655, 18.1258926, -67.1120605, 67.0918121
12: -52.8831444, 18.5211792, -52.8452034, 18.2166100, -69.4277725, 69.6806259
13: -35.8031158, 38.8361588, -35.7643890, 38.6984787, -74.5015945, 74.6005478
14: -78.5270004, 11.2542496, -78.4664917, 11.0831566, -89.6101532, 89.7207413
15: -30.3819065, 30.2812405, -30.3255081, 30.1316605, -60.5135651, 60.6067505
16: -46.4131279, 30.9026012, -46.2775269, 30.8640709, -77.2680969, 77.1700439
17: -77.9721832, 14.9222164, -77.9328003, 14.7058544, -92.6780396, 92.8550186
18: -45.8500061, 21.4043713, -45.8069687, 21.3255177, -67.1755219, 67.2113419
19: -34.5335960, 11.0067844, -34.5159492, 10.9817076, -45.5153046, 45.5227356
20: -30.6498394, 14.3576145, -30.6076164, 14.3092184, -44.9590569, 44.9652328
21: -42.7443237, 14.9643974, -42.7079315, 14.9298697, -57.6741943, 57.6723289
22: -43.3411598, 17.8255882, -43.2978401, 17.6328011, -60.9739609, 61.1234283
23: -34.4712753, 15.1847029, -34.4285545, 15.1638346, -49.6351089, 49.6132584
24: -36.5011597, 14.9239521, -36.4170723, 14.9011917, -51.4023514, 51.3410263
25: -35.5994339, 17.4198341, -35.5668449, 17.3543110, -52.9537430, 52.9866791
26: -53.6295853, 20.5402603, -53.5670547, 20.2673912, -73.8969727, 74.1073151
27: -36.3141747, 18.9507561, -36.2377777, 18.9352665, -55.2494431, 55.1885338
28: -33.3840294, 19.0494919, -33.3380814, 19.0216084, -52.4056396, 52.3875732
29: -45.0344810, 17.0618229, -44.9929810, 16.8893433, -61.9238243, 62.0548019
30: -42.9993324, 20.0807304, -42.8676376, 20.0505180, -63.0498505, 62.9483681
31: -42.3872833, 15.3562164, -42.3283234, 15.3186398, -57.7059250, 57.6845398
32: -38.5650253, 23.2862091, -38.5288391, 23.1980190, -61.7630463, 61.8150482
33: -48.9624863, 36.0181999, -48.8784485, 35.9742432, -84.9367294, 84.8966522
34: -47.2261391, 21.1532421, -47.1818161, 21.1244602, -68.3158264, 68.2996292
35: -41.7579880, 26.4552765, -41.7173004, 26.4310684, -67.7465515, 67.7201996
36: -42.4814529, 26.7901726, -42.4496803, 26.6420517, -68.2721558, 68.3908310
37: -66.9119720, 22.3766365, -66.8650665, 22.3238640, -86.7399139, 86.7256393
38: -52.6424561, 31.4197388, -52.5790710, 31.2717476, -82.1107178, 82.1965332
39: -60.3119125, 35.4950294, -60.2820816, 35.4570236, -95.7689362, 95.7771149
40: -53.6638184, 28.4241638, -53.5659294, 28.4070721, -82.0708923, 81.9900970
41: -39.1491089, 27.1975441, -39.1094627, 27.1592712, -66.3083801, 66.3070068
42: -32.5763245, 22.0448456, -32.5525589, 21.9938145, -54.5701370, 54.5974045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=152, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 48.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 53.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 103.84 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 103.84
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 103.84
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 103.84
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 103.84
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -43.1692200, 35.1856194, -42.9074783, 35.0699387, -78.2391586, 78.0930939
1: -23.4107494, 31.9828892, -23.2240467, 31.8783569, -55.2891083, 55.2069359
2: -18.9262486, 31.8727303, -18.6706333, 31.7558556, -50.6623917, 50.5222092
3: -19.1293564, 35.0846062, -18.8282604, 34.9181900, -53.6115112, 53.4631538
4: -23.5220070, 36.0109177, -23.3815498, 35.9187851, -59.4407921, 59.3924675
5: -21.3012886, 35.4243164, -20.9761772, 35.2434196, -55.9224014, 55.7651978
6: -42.2047462, 26.0257015, -42.0800896, 25.8492126, -68.0539551, 68.1057892
7: -30.5690079, 34.1120148, -30.1451187, 33.9094963, -63.8359222, 63.6100082
8: -29.1127872, 40.0761795, -28.8464241, 39.9276199, -69.0404053, 68.9226074
9: -24.3687897, 31.6875648, -24.2204475, 31.4924603, -54.6992798, 54.8099060
10: -45.7938194, 31.3811188, -45.5658798, 31.0113678, -76.8051910, 76.9469986
11: -48.9421158, 18.1487560, -48.7918663, 17.9448738, -66.8869934, 66.9406204
12: -52.6451340, 18.4941406, -52.3094673, 17.7675209, -68.7298737, 69.1363144
13: -35.7140770, 38.8092918, -35.5062752, 38.5264740, -74.2405548, 74.3155670
14: -78.3329697, 11.2416916, -77.9910278, 10.8027763, -89.1357422, 89.2327194
15: -30.2437744, 30.2564335, -29.9629154, 29.9025631, -60.1463394, 60.2193489
16: -46.3658295, 30.8048477, -46.1304779, 30.5994835, -76.9563141, 76.9260178
17: -77.8027267, 14.8982372, -77.5529556, 14.3687000, -92.1714249, 92.4511948
18: -45.7887917, 21.3813515, -45.6345863, 21.1910172, -66.9798126, 67.0159378
19: -34.5030136, 10.9974270, -34.4292107, 10.9231110, -45.4261246, 45.4266357
20: -30.6095924, 14.3467388, -30.4694519, 14.2368011, -44.8463936, 44.8161926
21: -42.7088356, 14.9495411, -42.6123428, 14.8343601, -57.5431976, 57.5618820
22: -43.1658096, 17.8056412, -42.8842659, 17.3562851, -60.5220947, 60.6899071
23: -34.4396095, 15.1584740, -34.3135567, 15.0774508, -49.5170593, 49.4720306
24: -36.4726181, 14.8967075, -36.2948837, 14.8174553, -51.2900734, 51.1915894
25: -35.5451279, 17.4003792, -35.4170151, 17.2344170, -52.7795448, 52.8173943
26: -53.3997879, 20.5128632, -53.0361710, 19.8788853, -73.2786713, 73.5490341
27: -36.2868423, 18.9169159, -36.0897446, 18.8307648, -55.1176071, 55.0066605
28: -33.3595924, 19.0156231, -33.2124405, 18.9168358, -52.2764282, 52.2280655
29: -44.9054260, 17.0490990, -44.6742554, 16.6368637, -61.5422897, 61.7233543
30: -42.9671707, 19.9847946, -42.6744118, 19.7949142, -62.7620850, 62.6592064
31: -42.3586884, 15.3411789, -42.2674103, 15.2314768, -57.5901642, 57.6085892
32: -38.5018806, 23.2619858, -38.3450851, 23.0769024, -61.5787811, 61.6070709
33: -48.9308548, 35.9800339, -48.6971359, 35.8408623, -84.7717133, 84.6771698
34: -47.2031822, 21.0901833, -47.0294342, 20.9602566, -68.1289368, 68.0807877
35: -41.7275581, 26.3966637, -41.5447540, 26.2905159, -67.5820541, 67.4716492
36: -42.4311600, 26.7702465, -42.2823448, 26.5495071, -68.1375580, 68.1943588
37: -66.8645248, 22.3504257, -66.6895905, 22.2193260, -86.5784607, 86.5133209
38: -52.5964241, 31.3820801, -52.3983040, 31.1297112, -81.9356537, 81.9637985
39: -60.2634392, 35.4618340, -60.1073227, 35.3269958, -95.5904388, 95.5691528
40: -53.6320801, 28.3318615, -53.3596878, 28.1990910, -81.8311691, 81.6915512
41: -39.1278267, 27.1405258, -39.0095673, 27.0000763, -66.1278992, 66.1500931
42: -32.5567284, 21.9966984, -32.4759064, 21.8326912, -54.3894196, 54.4726028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
time: 59.26 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
time: 55.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -43.1777496, 35.2143822, -42.9763527, 35.1336174, -78.3113708, 78.1907349
1: -23.4149857, 32.0098228, -23.2749367, 31.9370995, -55.3520851, 55.2847595
2: -18.9310589, 31.9004288, -18.7261353, 31.8170433, -50.7278900, 50.6054420
3: -19.1336765, 35.1121330, -18.8791084, 34.9813385, -53.6752472, 53.5417099
4: -23.5255966, 36.0278511, -23.4143353, 35.9586716, -59.4842682, 59.4421844
5: -21.3048630, 35.4547348, -21.0291862, 35.3130951, -55.9936371, 55.8495712
6: -42.2124786, 26.0591202, -42.1118126, 25.9294605, -68.1419373, 68.1709290
7: -30.5774059, 34.1678162, -30.2375412, 34.0303001, -63.9620514, 63.7561493
8: -29.1166744, 40.1137848, -28.9112530, 40.0108566, -69.1275330, 69.0250397
9: -24.3738060, 31.6992435, -24.2439251, 31.5253067, -54.7368622, 54.8446579
10: -45.8051033, 31.3935566, -45.6051292, 31.0627556, -76.8678589, 76.9986877
11: -48.9593964, 18.1677094, -48.8299484, 18.0077019, -66.9671021, 66.9976578
12: -52.7173347, 18.5010834, -52.4644432, 17.8955727, -68.9295197, 69.2918243
13: -35.7467728, 38.8202133, -35.5885658, 38.5717316, -74.3185043, 74.4087830
14: -78.3891983, 11.2470493, -78.1214371, 10.8714561, -89.2606506, 89.3684845
15: -30.3014660, 30.2674828, -30.1025715, 29.9652367, -60.2667007, 60.3700562
16: -46.3802719, 30.8748055, -46.2153015, 30.7537766, -77.1244736, 77.0804443
17: -77.8647614, 14.9079380, -77.6859131, 14.4682865, -92.3330460, 92.5938492
18: -45.8106575, 21.3889732, -45.6896515, 21.2207260, -67.0313873, 67.0786285
19: -34.5122452, 10.9977121, -34.4472427, 10.9372168, -45.4494629, 45.4449539
20: -30.6238575, 14.3498535, -30.5124149, 14.2591610, -44.8830185, 44.8622665
21: -42.7208633, 14.9529753, -42.6343842, 14.8592672, -57.5801315, 57.5873604
22: -43.2462769, 17.8137817, -43.0562286, 17.4551735, -60.7014503, 60.8700104
23: -34.4511032, 15.1634741, -34.3453064, 15.0956707, -49.5467758, 49.5087814
24: -36.4848251, 14.9032555, -36.3292618, 14.8377171, -51.3225403, 51.2325172
25: -35.5746422, 17.4066010, -35.4830399, 17.2801628, -52.8548050, 52.8896408
26: -53.4849014, 20.5203495, -53.2180252, 19.9976158, -73.4825134, 73.7383728
27: -36.2988167, 18.9253693, -36.1213150, 18.8560810, -55.1548996, 55.0466843
28: -33.3687286, 19.0193977, -33.2436523, 18.9347267, -52.3034554, 52.2630501
29: -44.9697609, 17.0525150, -44.8140259, 16.7217979, -61.6915588, 61.8665390
30: -42.9802475, 20.0059834, -42.7073975, 19.8544312, -62.8346786, 62.7133789
31: -42.3674164, 15.3468771, -42.2834015, 15.2606335, -57.6280518, 57.6302795
32: -38.5285034, 23.2698612, -38.4092865, 23.1128273, -61.6413307, 61.6791458
33: -48.9398308, 35.9930649, -48.7323151, 35.8835220, -84.8233490, 84.7253799
34: -47.2091675, 21.0997505, -47.0547638, 20.9832573, -68.1579437, 68.1165619
35: -41.7361946, 26.4041100, -41.5711746, 26.3046818, -67.6014175, 67.5065842
36: -42.4596024, 26.7752972, -42.3520927, 26.6022491, -68.2143860, 68.2659760
37: -66.8813782, 22.3588314, -66.7391663, 22.2483864, -86.6280212, 86.5740051
38: -52.6261406, 31.3904610, -52.4769821, 31.1819096, -82.0072632, 82.0416183
39: -60.2735939, 35.4744225, -60.1381836, 35.3648720, -95.6384659, 95.6126099
40: -53.6425362, 28.3781166, -53.4188347, 28.2940464, -81.9365845, 81.7969513
41: -39.1364288, 27.1636028, -39.0430756, 27.0581169, -66.1945496, 66.2066803
42: -32.5646973, 22.0147285, -32.5010033, 21.8862476, -54.4509430, 54.5157318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
time: 59.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
time: 57.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.1928940, 35.2033768, -43.0286407, 35.1406288, -78.3335266, 78.2320175
1: -23.4218903, 31.9970627, -23.2910671, 31.9430122, -55.3649025, 55.2881317
2: -18.9398956, 31.9174995, -18.7790146, 31.8652954, -50.7841263, 50.6765251
3: -19.1419621, 35.1474648, -18.9721985, 35.0775032, -53.7725372, 53.6810837
4: -23.5423241, 36.0229950, -23.4597359, 35.9770432, -59.5193672, 59.4827309
5: -21.3154087, 35.4891472, -21.1370583, 35.4171600, -56.1042557, 55.9979935
6: -42.2150116, 26.0712051, -42.1480370, 25.9768372, -68.1918488, 68.2192383
7: -30.5873375, 34.1741371, -30.3134346, 34.0757828, -64.0149384, 63.8498764
8: -29.1223736, 40.1107597, -28.9459190, 40.0348816, -69.1572571, 69.0566788
9: -24.4294415, 31.6964378, -24.3768539, 31.6251450, -54.9339294, 54.9714355
10: -45.9065094, 31.3992195, -45.8312378, 31.2494164, -77.1559296, 77.2304535
11: -48.9661942, 18.1546745, -48.8601761, 18.0314484, -66.9976425, 67.0148468
12: -52.8086395, 18.5137215, -52.6852875, 18.0877304, -69.2249985, 69.5134583
13: -35.7619324, 38.8238907, -35.6627769, 38.6490707, -74.4110031, 74.4866638
14: -78.4677200, 11.2483883, -78.3288956, 11.0129261, -89.4806442, 89.5772858
15: -30.3126469, 30.2683372, -30.1602612, 30.0625076, -60.3751526, 60.4285965
16: -46.3961296, 30.8198586, -46.1865883, 30.6808853, -77.0680923, 76.9968262
17: -77.9074707, 14.9106960, -77.7940063, 14.6018963, -92.5093689, 92.7047043
18: -45.8268127, 21.3953609, -45.7490082, 21.2925854, -67.1194000, 67.1443710
19: -34.5232391, 11.0021105, -34.4952850, 10.9582729, -45.4815140, 45.4973946
20: -30.6341400, 14.3540668, -30.5614948, 14.2852459, -44.9193878, 44.9155617
21: -42.7308044, 14.9550743, -42.6823502, 14.8903913, -57.6211967, 57.6374245
22: -43.2535820, 17.8159981, -43.1080856, 17.5304661, -60.7840500, 60.9240837
23: -34.4588394, 15.1774235, -34.3945580, 15.1400795, -49.5989189, 49.5719833
24: -36.4872513, 14.9151649, -36.3781204, 14.8758001, -51.3630524, 51.2932854
25: -35.5658035, 17.4126511, -35.4907379, 17.3064194, -52.8722229, 52.9033890
26: -53.5411835, 20.5313683, -53.3773499, 20.1453915, -73.6865768, 73.9087219
27: -36.3006058, 18.9363384, -36.2023201, 18.8969650, -55.1975708, 55.1386566
28: -33.3741608, 19.0451374, -33.3051376, 19.0023746, -52.3765335, 52.3502731
29: -44.9660034, 17.0578499, -44.8437767, 16.8031387, -61.7691422, 61.9016266
30: -42.9839096, 20.0506973, -42.8291740, 19.9761791, -62.9600906, 62.8798714
31: -42.3774376, 15.3449478, -42.3092613, 15.2761612, -57.6535988, 57.6542091
32: -38.5365334, 23.2774773, -38.4606628, 23.1601620, -61.6966934, 61.7381401
33: -48.9524956, 36.0038071, -48.8408508, 35.9288940, -84.8813934, 84.8446579
34: -47.2168274, 21.1424179, -47.1486397, 21.0987377, -68.2807770, 68.2553101
35: -41.7435608, 26.4469376, -41.6773949, 26.4147835, -67.7193298, 67.6724548
36: -42.4463654, 26.7846909, -42.3639297, 26.5877399, -68.1867523, 68.2999039
37: -66.8923187, 22.3653564, -66.8088379, 22.2880821, -86.6808014, 86.6561584
38: -52.6086884, 31.4101372, -52.4924736, 31.2166729, -82.0312347, 82.1033173
39: -60.2986221, 35.4812164, -60.2437859, 35.4165459, -95.7151642, 95.7250061
40: -53.6514053, 28.3721447, -53.5022430, 28.2981682, -81.9495697, 81.8743896
41: -39.1393661, 27.1704559, -39.0732727, 27.0916862, -66.2310486, 66.2437286
42: -32.5673828, 22.0230465, -32.5252380, 21.9316120, -54.4989929, 54.5482864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
time: 56.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
time: 54.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.2013931, 35.2324104, -43.0974731, 35.2042923, -78.4056854, 78.3298798
1: -23.4261017, 32.0241356, -23.3420181, 32.0015335, -55.4276352, 55.3661537
2: -18.9446697, 31.9452248, -18.8343716, 31.9263420, -50.8495026, 50.7596054
3: -19.1462708, 35.1750298, -19.0228386, 35.1404724, -53.8360062, 53.7594147
4: -23.5460835, 36.0399170, -23.4924164, 36.0168915, -59.5629730, 59.5323334
5: -21.3189754, 35.5195808, -21.1899529, 35.4866409, -56.1753387, 56.0822906
6: -42.2227631, 26.1046009, -42.1796837, 26.0570507, -68.2798157, 68.2842865
7: -30.5957050, 34.2299423, -30.4057884, 34.1958237, -64.1402283, 63.9959717
8: -29.1262493, 40.1483917, -29.0106735, 40.1179199, -69.2441711, 69.1590652
9: -24.4344368, 31.7080345, -24.4002457, 31.6578102, -54.9714966, 55.0060234
10: -45.9177856, 31.4115601, -45.8702240, 31.3004456, -77.2182312, 77.2817841
11: -48.9834366, 18.1735229, -48.8980827, 18.0950184, -67.0784531, 67.0716095
12: -52.8808784, 18.5206528, -52.8398895, 18.2154369, -69.4243240, 69.6686096
13: -35.7948685, 38.8347778, -35.7445755, 38.6953506, -74.4902191, 74.5793533
14: -78.5240479, 11.2536449, -78.4591522, 11.0816383, -89.6056824, 89.7127991
15: -30.3705349, 30.2793865, -30.2992210, 30.1273155, -60.4978485, 60.5786057
16: -46.4105492, 30.8896294, -46.2714081, 30.8350105, -77.2359924, 77.1510010
17: -77.9695663, 14.9203472, -77.9266968, 14.7014904, -92.6710587, 92.8470459
18: -45.8487091, 21.4029465, -45.8039398, 21.3221169, -67.1708221, 67.2068863
19: -34.5324516, 11.0023708, -34.5132751, 10.9718676, -45.5043182, 45.5156479
20: -30.6484718, 14.3571739, -30.6044521, 14.3075581, -44.9560318, 44.9616241
21: -42.7428017, 14.9583950, -42.7043915, 14.9151592, -57.6579590, 57.6627884
22: -43.3341103, 17.8241348, -43.2812004, 17.6293678, -60.9634781, 61.1053352
23: -34.4702835, 15.1824112, -34.4263077, 15.1582146, -49.6284981, 49.6087189
24: -36.4994125, 14.9217625, -36.4128799, 14.8958941, -51.3953056, 51.3346405
25: -35.5955429, 17.4188843, -35.5570526, 17.3520908, -52.9476318, 52.9759369
26: -53.6263657, 20.5388145, -53.5588455, 20.2639256, -73.8902893, 74.0976562
27: -36.3125763, 18.9454651, -36.2340660, 18.9229088, -55.2354851, 55.1795311
28: -33.3833046, 19.0489006, -33.3363571, 19.0202427, -52.4035492, 52.3852577
29: -45.0304375, 17.0612526, -44.9833946, 16.8880463, -61.9184837, 62.0446472
30: -42.9969749, 20.0719261, -42.8621559, 20.0346546, -63.0316315, 62.9340820
31: -42.3861542, 15.3506489, -42.3255959, 15.3050337, -57.6911888, 57.6762466
32: -38.5631561, 23.2852859, -38.5245819, 23.1958961, -61.7590523, 61.8098679
33: -48.9614182, 36.0168915, -48.8758965, 35.9712753, -84.9326935, 84.8927917
34: -47.2227859, 21.1520348, -47.1738930, 21.1216469, -68.3097229, 68.2910690
35: -41.7521286, 26.4543419, -41.7035942, 26.4288883, -67.7386475, 67.7072449
36: -42.4748154, 26.7897320, -42.4338226, 26.6410275, -68.2641602, 68.3718262
37: -66.9091187, 22.3736687, -66.8583450, 22.3171501, -86.7303314, 86.7168579
38: -52.6384659, 31.4185390, -52.5704918, 31.2689381, -82.1028900, 82.1802521
39: -60.3087692, 35.4937859, -60.2746048, 35.4544411, -95.7632141, 95.7683868
40: -53.6618500, 28.4189930, -53.5613403, 28.3933105, -82.0551605, 81.9803314
41: -39.1479454, 27.1935062, -39.1067200, 27.1496449, -66.2975922, 66.3002243
42: -32.5753517, 22.0412712, -32.5502853, 21.9850864, -54.5604401, 54.5915565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=151, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4915079
time: 57.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.5124481
time: 55.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 115.52 seconds
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4915079
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.52
Output dim: 8, lower bound: -37.4690362, upper bound: 37.5124481

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -43.0805893, 35.0967064, -42.8946228, 35.0283928, -78.1089783, 77.9913330
1: -23.3570480, 31.9295273, -23.2180157, 31.8544121, -55.2114601, 55.1475449
2: -18.8806248, 31.8296432, -18.6652508, 31.7370586, -50.5956383, 50.4736023
3: -19.0989075, 35.0604095, -18.8209839, 34.9097900, -53.5681152, 53.4312134
4: -23.4525318, 35.9313927, -23.3745193, 35.8829422, -59.3354721, 59.3059120
5: -21.2663651, 35.3649178, -20.9711609, 35.2188225, -55.8534393, 55.7002029
6: -42.1526604, 25.9426060, -42.0726662, 25.8136749, -67.9663391, 68.0152740
7: -30.5133724, 34.0490723, -30.1381493, 33.8822327, -63.7386932, 63.5388260
8: -29.0490608, 39.9930496, -28.8416348, 39.8917694, -68.9408264, 68.8346863
9: -24.3290119, 31.6384087, -24.2148151, 31.4718170, -54.6127586, 54.7542877
10: -45.7465286, 31.3394318, -45.5528145, 31.0009575, -76.7474823, 76.8922424
11: -48.8419800, 18.0570488, -48.7507553, 17.9396629, -66.7816467, 66.8078003
12: -52.5951614, 18.4585686, -52.2925415, 17.7626305, -68.6714478, 69.0514984
13: -35.6510162, 38.7785263, -35.4908295, 38.5171280, -74.1681442, 74.2693558
14: -78.2697525, 11.1896248, -77.9683380, 10.7979546, -89.0677032, 89.1579590
15: -30.1797962, 30.2228317, -29.9418526, 29.8942013, -60.0739975, 60.1646843
16: -46.2868118, 30.7231483, -46.1195297, 30.5646801, -76.8391724, 76.8324585
17: -77.6886520, 14.7695332, -77.5090790, 14.3604755, -92.0491257, 92.2786102
18: -45.7724304, 21.3072739, -45.6253471, 21.1632938, -66.9357224, 66.9326172
19: -34.4262352, 10.9306202, -34.3981323, 10.9195995, -45.3458328, 45.3287506
20: -30.5614243, 14.3048019, -30.4512711, 14.2335014, -44.7949257, 44.7560730
21: -42.6055450, 14.8573847, -42.5708771, 14.8309708, -57.4365158, 57.4282608
22: -43.0351181, 17.7020073, -42.8288002, 17.3519421, -60.3870621, 60.5308075
23: -34.3477249, 15.0743818, -34.2737885, 15.0732756, -49.4210014, 49.3481712
24: -36.3846664, 14.8433666, -36.2574387, 14.8149004, -51.1995659, 51.1008072
25: -35.4377937, 17.3068104, -35.3689651, 17.2283001, -52.6660919, 52.6757736
26: -53.3270149, 20.4424438, -53.0085487, 19.8744621, -73.2014771, 73.4509888
27: -36.2616577, 18.8678207, -36.0794258, 18.8192749, -55.0809326, 54.9472466
28: -33.2857895, 18.9426498, -33.1812515, 18.9127235, -52.1985130, 52.1239014
29: -44.7427673, 16.9177341, -44.6038513, 16.6323566, -61.3751221, 61.5215836
30: -42.8549347, 19.8934803, -42.6265411, 19.7895241, -62.6444588, 62.5200195
31: -42.2866478, 15.2810678, -42.2383385, 15.2273149, -57.5139618, 57.5194054
32: -38.4537125, 23.2093143, -38.3346176, 23.0559673, -61.5096817, 61.5439301
33: -48.8562355, 35.9615173, -48.6797523, 35.8339462, -84.6901855, 84.6412659
34: -47.1645432, 21.0618515, -47.0233002, 20.9532032, -68.0828400, 68.0459442
35: -41.6694832, 26.3682213, -41.5260887, 26.2841682, -67.5129013, 67.4155579
36: -42.3887177, 26.7453671, -42.2749557, 26.5431042, -68.0777893, 68.1593018
37: -66.7987213, 22.2881145, -66.6755066, 22.1916504, -86.4613953, 86.4268036
38: -52.5259171, 31.2936211, -52.3874664, 31.0919266, -81.7971039, 81.8475037
39: -60.1782112, 35.4182816, -60.0947876, 35.3084793, -95.4866943, 95.5130692
40: -53.5090981, 28.1984329, -53.3474388, 28.1362934, -81.6453934, 81.5458679
41: -39.0716209, 27.0769386, -39.0027161, 26.9732265, -66.0448456, 66.0796509
42: -32.5201492, 21.9738617, -32.4651337, 21.8247299, -54.3448792, 54.4389954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3950150, upper bound: 37.4893271
time: 51.09 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3950150, upper bound: 37.4893271
time: 57.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.1668015, 35.1800766, -42.9062576, 35.0669861, -78.2337875, 78.0863342
1: -23.4098587, 31.9796219, -23.2236023, 31.8767471, -55.2866058, 55.2032242
2: -18.9252567, 31.8696899, -18.6701469, 31.7541924, -50.6610413, 50.5175629
3: -19.1213608, 35.0821228, -18.8243370, 34.9169312, -53.6149139, 53.4557800
4: -23.5195713, 36.0026398, -23.3802929, 35.9141846, -59.4337540, 59.3829346
5: -21.2998905, 35.4207344, -20.9754658, 35.2416229, -55.9243469, 55.7567253
6: -42.2030907, 26.0194397, -42.0792847, 25.8462543, -68.0493469, 68.0987244
7: -30.5676594, 34.1083527, -30.1444397, 33.9076614, -63.8395233, 63.6003571
8: -29.1115513, 40.0698929, -28.8457794, 39.9244843, -69.0360336, 68.9156723
9: -24.3672161, 31.6851044, -24.2196693, 31.4912148, -54.7135010, 54.7959366
10: -45.7828369, 31.3790684, -45.5601959, 31.0102921, -76.7931290, 76.9392624
11: -48.9324341, 18.1468315, -48.7865906, 17.9438896, -66.8763275, 66.9334259
12: -52.6404762, 18.4924259, -52.3071213, 17.7666779, -68.7147369, 69.1484070
13: -35.6946793, 38.8065338, -35.4967041, 38.5251007, -74.2197800, 74.3032379
14: -78.3287811, 11.2402878, -77.9888916, 10.8020573, -89.1308365, 89.2291794
15: -30.2393932, 30.2543640, -29.9608154, 29.9014950, -60.1408882, 60.2151794
16: -46.3638000, 30.7993546, -46.1294632, 30.5967216, -76.9532166, 76.9184875
17: -77.7976456, 14.8964596, -77.5504150, 14.3677692, -92.1654129, 92.4468765
18: -45.7849197, 21.3570404, -45.6326218, 21.1788826, -66.9638062, 66.9896622
19: -34.4983902, 10.9967413, -34.4269180, 10.9227419, -45.4211311, 45.4236603
20: -30.6065140, 14.3461037, -30.4679070, 14.2364693, -44.8429832, 44.8140106
21: -42.7028618, 14.9487171, -42.6092606, 14.8339539, -57.5368156, 57.5579758
22: -43.1583748, 17.8047695, -42.8800774, 17.3558502, -60.5142250, 60.6848450
23: -34.4339790, 15.1568356, -34.3100891, 15.0766344, -49.5106125, 49.4669266
24: -36.4679871, 14.8962822, -36.2925491, 14.8172417, -51.2852287, 51.1888313
25: -35.5393066, 17.3988609, -35.4135895, 17.2336140, -52.7729187, 52.8124504
26: -53.3955078, 20.5116882, -53.0340042, 19.8782864, -73.2737961, 73.5456924
27: -36.2842560, 18.9017105, -36.0884171, 18.8233032, -55.1075592, 54.9901276
28: -33.3553619, 19.0141258, -33.2103691, 18.9160957, -52.2714577, 52.2244949
29: -44.8979874, 17.0460453, -44.6702614, 16.6352997, -61.5332870, 61.7163086
30: -42.9593887, 19.9826431, -42.6691093, 19.7938423, -62.7532310, 62.6517525
31: -42.3538818, 15.3405085, -42.2649956, 15.2311516, -57.5850334, 57.6055031
32: -38.4983673, 23.2596302, -38.3432693, 23.0757351, -61.5741043, 61.6028976
33: -48.9134789, 35.9782104, -48.6885834, 35.8399353, -84.7534180, 84.6667938
34: -47.2005310, 21.0876617, -47.0280914, 20.9590263, -68.1250610, 68.0764313
35: -41.7188339, 26.3945694, -41.5404358, 26.2894783, -67.5658188, 67.4654312
36: -42.4281616, 26.7653332, -42.2808075, 26.5470085, -68.1318359, 68.1764984
37: -66.8613739, 22.3376999, -66.6879959, 22.2127724, -86.5734558, 86.4594879
38: -52.5903320, 31.3614063, -52.3952255, 31.1190567, -81.9261780, 81.9181061
39: -60.2579575, 35.4596138, -60.1045532, 35.3259048, -95.5838623, 95.5641632
40: -53.6286354, 28.3215618, -53.3579559, 28.1940823, -81.8227158, 81.6795197
41: -39.1265869, 27.1359138, -39.0089836, 26.9977703, -66.1243591, 66.1448975
42: -32.5481186, 21.9940815, -32.4714813, 21.8313637, -54.3794823, 54.4655609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 665

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3950150, upper bound: 37.5102708
time: 49.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3950150, upper bound: 37.5102708
time: 44.25 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -43.0891571, 35.1254501, -42.9634781, 35.0920410, -78.1811981, 78.0889282
1: -23.3612785, 31.9564857, -23.2689362, 31.9131489, -55.2744293, 55.2254219
2: -18.8854218, 31.8573284, -18.7207355, 31.7982464, -50.6611633, 50.5568123
3: -19.1032295, 35.0879478, -18.8718414, 34.9729233, -53.6318512, 53.5097618
4: -23.4561424, 35.9482880, -23.4073029, 35.9228210, -59.3765106, 59.3555908
5: -21.2699566, 35.3953629, -21.0241737, 35.2885094, -55.9247437, 55.7846107
6: -42.1604233, 25.9760342, -42.1043968, 25.8939037, -68.0543289, 68.0804291
7: -30.5218124, 34.1048698, -30.2306137, 34.0030556, -63.8648300, 63.6850128
8: -29.0529346, 40.0306702, -28.9064808, 39.9749794, -69.0279160, 68.9371490
9: -24.3340206, 31.6500492, -24.2382870, 31.5046196, -54.6503639, 54.7890739
10: -45.7577972, 31.3518600, -45.5920639, 31.0523148, -76.8101120, 76.9439240
11: -48.8592300, 18.0760632, -48.7888908, 18.0023994, -66.8616333, 66.8649521
12: -52.6674423, 18.4655457, -52.4474564, 17.8906937, -68.8711090, 69.2070160
13: -35.6836929, 38.7894516, -35.5731049, 38.5624352, -74.2461243, 74.3625565
14: -78.3259583, 11.1949387, -78.0987701, 10.8666229, -89.1925812, 89.2937088
15: -30.2374916, 30.2338829, -30.0814972, 29.9568787, -60.1943703, 60.3153801
16: -46.3012810, 30.7931099, -46.2044106, 30.7189655, -77.0073395, 76.9868774
17: -77.7506790, 14.7792530, -77.6420288, 14.4600983, -92.2107773, 92.4212799
18: -45.7943115, 21.3148060, -45.6803513, 21.1930008, -66.9873123, 66.9951553
19: -34.4354439, 10.9308882, -34.4161530, 10.9337502, -45.3691940, 45.3470421
20: -30.5756950, 14.3079090, -30.4942226, 14.2558479, -44.8315430, 44.8021317
21: -42.6175613, 14.8608322, -42.5929337, 14.8558702, -57.4734306, 57.4537659
22: -43.1155472, 17.7101364, -43.0007591, 17.4508247, -60.5663719, 60.7108955
23: -34.3591766, 15.0793972, -34.3055687, 15.0914955, -49.4506721, 49.3849640
24: -36.3968468, 14.8499165, -36.2918243, 14.8351545, -51.2320023, 51.1417389
25: -35.4672928, 17.3130341, -35.4349899, 17.2740517, -52.7413445, 52.7480240
26: -53.4121628, 20.4499016, -53.1904182, 19.9931526, -73.4053192, 73.6403198
27: -36.2736282, 18.8762360, -36.1110458, 18.8445129, -55.1181412, 54.9872818
28: -33.2949257, 18.9464207, -33.2124557, 18.9305840, -52.2255096, 52.1588745
29: -44.8071098, 16.9211960, -44.7436180, 16.7172813, -61.5243912, 61.6648140
30: -42.8680038, 19.9146767, -42.6595535, 19.8487797, -62.7167816, 62.5742302
31: -42.2953835, 15.2868147, -42.2543488, 15.2564468, -57.5518303, 57.5411644
32: -38.4803505, 23.2171936, -38.3987732, 23.0918980, -61.5722504, 61.6159668
33: -48.8652115, 35.9744797, -48.7149544, 35.8765717, -84.7417831, 84.6894379
34: -47.1704865, 21.0714111, -47.0486031, 20.9762039, -68.1118774, 68.0817032
35: -41.6780853, 26.3756676, -41.5525322, 26.2983017, -67.5322800, 67.4505539
36: -42.4172325, 26.7504158, -42.3446960, 26.5957966, -68.1547623, 68.2308655
37: -66.8155441, 22.2964840, -66.7250671, 22.2206306, -86.5109863, 86.4874725
38: -52.5557022, 31.3019505, -52.4661713, 31.1441345, -81.8687286, 81.9252548
39: -60.1883354, 35.4308624, -60.1256752, 35.3463554, -95.5346909, 95.5565338
40: -53.5195847, 28.2446651, -53.4066353, 28.2312202, -81.7508087, 81.6512985
41: -39.0801926, 27.0999680, -39.0362167, 27.0312233, -66.1114197, 66.1361847
42: -32.5281143, 21.9918861, -32.4902039, 21.8782578, -54.4063721, 54.4820900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3858720, upper bound: 37.4865876
time: 62.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3858720, upper bound: 37.4865876
time: 56.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.1753273, 35.2088242, -42.9751015, 35.1306915, -78.3060150, 78.1839294
1: -23.4140682, 32.0065536, -23.2744789, 31.9354706, -55.3495407, 55.2810326
2: -18.9300671, 31.8973770, -18.7256222, 31.8153725, -50.7265244, 50.6008148
3: -19.1257076, 35.1096497, -18.8751869, 34.9800644, -53.6786041, 53.5343208
4: -23.5231571, 36.0195961, -23.4130745, 35.9540558, -59.4772110, 59.4326706
5: -21.3034916, 35.4511986, -21.0284786, 35.3113136, -55.9956284, 55.8411102
6: -42.2108307, 26.0528622, -42.1110001, 25.9265003, -68.1373291, 68.1638641
7: -30.5760536, 34.1641541, -30.2368374, 34.0284538, -63.9656525, 63.7464905
8: -29.1154079, 40.1075630, -28.9106197, 40.0077591, -69.1231689, 69.0181808
9: -24.3722286, 31.6967697, -24.2431431, 31.5240440, -54.7511063, 54.8307152
10: -45.7941055, 31.3914986, -45.5994797, 31.0616856, -76.8557892, 76.9909821
11: -48.9497070, 18.1657906, -48.8246765, 18.0067234, -66.9564285, 66.9904633
12: -52.7127266, 18.4994030, -52.4620895, 17.8947430, -68.9143982, 69.3039474
13: -35.7273903, 38.8174477, -35.5790176, 38.5703430, -74.2977295, 74.3964691
14: -78.3850098, 11.2455940, -78.1193085, 10.8707314, -89.2557373, 89.3648987
15: -30.2971535, 30.2654419, -30.1004734, 29.9641819, -60.2613373, 60.3659134
16: -46.3782349, 30.8692989, -46.2143097, 30.7510071, -77.1214066, 77.0728989
17: -77.8596573, 14.9061775, -77.6833801, 14.4674015, -92.3270569, 92.5895538
18: -45.8067932, 21.3646774, -45.6876450, 21.2086296, -67.0154266, 67.0523224
19: -34.5076180, 10.9970198, -34.4449387, 10.9368753, -45.4444923, 45.4419594
20: -30.6207752, 14.3492088, -30.5108604, 14.2588348, -44.8796082, 44.8600693
21: -42.7149200, 14.9521408, -42.6312943, 14.8588381, -57.5737572, 57.5834351
22: -43.2388229, 17.8129005, -43.0520248, 17.4547424, -60.6935654, 60.8649254
23: -34.4454536, 15.1618319, -34.3418427, 15.0948725, -49.5403252, 49.5036736
24: -36.4802017, 14.9028282, -36.3269119, 14.8375092, -51.3177109, 51.2297401
25: -35.5687943, 17.4050713, -35.4796448, 17.2793560, -52.8481522, 52.8847160
26: -53.4805984, 20.5191422, -53.2158813, 19.9970264, -73.4776230, 73.7350235
27: -36.2962418, 18.9101524, -36.1199875, 18.8486137, -55.1448555, 55.0301399
28: -33.3645020, 19.0179024, -33.2415619, 18.9339619, -52.2984619, 52.2594643
29: -44.9623184, 17.0494900, -44.8100204, 16.7202320, -61.6825485, 61.8595123
30: -42.9725037, 20.0037937, -42.7021332, 19.8533478, -62.8258514, 62.7059250
31: -42.3626022, 15.3462315, -42.2810020, 15.2602940, -57.6228943, 57.6272354
32: -38.5249939, 23.2675209, -38.4074593, 23.1116409, -61.6366348, 61.6749802
33: -48.9224281, 35.9911423, -48.7237549, 35.8826065, -84.8050385, 84.7148972
34: -47.2065277, 21.0972519, -47.0534401, 20.9820137, -68.1540909, 68.1122208
35: -41.7274742, 26.4020042, -41.5668793, 26.3035927, -67.5852051, 67.5003815
36: -42.4566116, 26.7704201, -42.3505859, 26.5996933, -68.2086639, 68.2481003
37: -66.8782425, 22.3461132, -66.7375336, 22.2417870, -86.6230545, 86.5202026
38: -52.6201057, 31.3697624, -52.4739418, 31.1712685, -81.9978027, 81.9958954
39: -60.2680969, 35.4721375, -60.1354141, 35.3637962, -95.6318970, 95.6075516
40: -53.6391335, 28.3678493, -53.4171295, 28.2890148, -81.9281464, 81.7849808
41: -39.1351700, 27.1589413, -39.0424690, 27.0558014, -66.1909714, 66.2014084
42: -32.5561066, 22.0121155, -32.4965744, 21.8848858, -54.4409943, 54.5086899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3858720, upper bound: 37.5075426
time: 50.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3858720, upper bound: 37.5075426
time: 48.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -43.1042519, 35.1144600, -43.0157967, 35.0990601, -78.2033081, 78.1302567
1: -23.3682003, 31.9437065, -23.2850685, 31.9190350, -55.2872353, 55.2287750
2: -18.8942661, 31.8744202, -18.7736340, 31.8464928, -50.7173843, 50.6279373
3: -19.1115227, 35.1232872, -18.9649353, 35.0691185, -53.7291107, 53.6491280
4: -23.4728775, 35.9434662, -23.4527168, 35.9411964, -59.4140739, 59.3961830
5: -21.2804813, 35.4297791, -21.1320457, 35.3925629, -56.0353699, 55.9330711
6: -42.1629639, 25.9881268, -42.1406136, 25.9413528, -68.1043167, 68.1287384
7: -30.5317554, 34.1111870, -30.3064995, 34.0485191, -63.9177094, 63.7787170
8: -29.0586605, 40.0276604, -28.9411469, 39.9990311, -69.0576935, 68.9688110
9: -24.3896618, 31.6472569, -24.3712273, 31.6045170, -54.8474274, 54.9158554
10: -45.8592262, 31.3575382, -45.8181610, 31.2389793, -77.0982056, 77.1756973
11: -48.8660278, 18.0630112, -48.8191147, 18.0262299, -66.8922577, 66.8821259
12: -52.7586937, 18.4782009, -52.6683159, 18.0827751, -69.1665726, 69.4286499
13: -35.6989365, 38.7931557, -35.6473389, 38.6397705, -74.3387070, 74.4404907
14: -78.4044647, 11.1963310, -78.3062057, 11.0081863, -89.4126511, 89.5025330
15: -30.2486973, 30.2347450, -30.1392155, 30.0541687, -60.3028641, 60.3739624
16: -46.3171234, 30.7381344, -46.1756592, 30.6460991, -76.9509964, 76.9032440
17: -77.7934113, 14.7819405, -77.7501373, 14.5936852, -92.3871002, 92.5320740
18: -45.8104782, 21.3212242, -45.7397232, 21.2648849, -67.0753632, 67.0609436
19: -34.4464417, 10.9352884, -34.4641724, 10.9547815, -45.4012222, 45.3994598
20: -30.5860100, 14.3121157, -30.5432968, 14.2819452, -44.8679543, 44.8554115
21: -42.6274681, 14.8629322, -42.6408615, 14.8870287, -57.5144958, 57.5037918
22: -43.1229095, 17.7123795, -43.0526352, 17.5261269, -60.6490364, 60.7650146
23: -34.3669395, 15.0933418, -34.3548088, 15.1358767, -49.5028152, 49.4481506
24: -36.3992729, 14.8618546, -36.3406410, 14.8732262, -51.2724991, 51.2024956
25: -35.4584427, 17.3190918, -35.4426804, 17.3003044, -52.7587471, 52.7617722
26: -53.4684448, 20.4609566, -53.3497162, 20.1409435, -73.6093903, 73.8106689
27: -36.2754097, 18.8872395, -36.1920204, 18.8854752, -55.1608849, 55.0792618
28: -33.3003693, 18.9721489, -33.2739563, 18.9982262, -52.2985954, 52.2461052
29: -44.8033905, 16.9265213, -44.7734108, 16.7986450, -61.6020355, 61.6999321
30: -42.8716545, 19.9593868, -42.7813263, 19.9707909, -62.8424454, 62.7407150
31: -42.3053894, 15.2848501, -42.2802391, 15.2719812, -57.5773697, 57.5650902
32: -38.4883766, 23.2248154, -38.4502296, 23.1392384, -61.6276169, 61.6750450
33: -48.8778152, 35.9852982, -48.8234787, 35.9219894, -84.7998047, 84.8087769
34: -47.1781960, 21.1141014, -47.1424713, 21.0917492, -68.2347717, 68.2205200
35: -41.6855087, 26.4184990, -41.6587486, 26.4083977, -67.6501694, 67.6163788
36: -42.4040375, 26.7597771, -42.3565826, 26.5813236, -68.1269836, 68.2649307
37: -66.8264999, 22.3029823, -66.7947006, 22.2603416, -86.5637054, 86.5695801
38: -52.5381775, 31.3216648, -52.4816551, 31.1789379, -81.8927765, 81.9870148
39: -60.2134132, 35.4376488, -60.2312927, 35.3980217, -95.6114349, 95.6689453
40: -53.5284424, 28.2387505, -53.4900436, 28.2353325, -81.7637787, 81.7287903
41: -39.0831566, 27.1068573, -39.0663834, 27.0648079, -66.1479645, 66.1732407
42: -32.5307922, 22.0001945, -32.5144501, 21.9236107, -54.4544029, 54.5146446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3858720, upper bound: 37.4865876
time: 52.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4218684, upper bound: 37.4865876
time: 41.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.1904716, 35.1978149, -43.0274124, 35.1377029, -78.3281708, 78.2252274
1: -23.4210129, 31.9938126, -23.2906170, 31.9413738, -55.3623886, 55.2844315
2: -18.9389286, 31.9144650, -18.7785072, 31.8636074, -50.7827835, 50.6718864
3: -19.1339836, 35.1450119, -18.9682770, 35.0762634, -53.7758560, 53.6736870
4: -23.5398827, 36.0147133, -23.4584808, 35.9724121, -59.5122948, 59.4731941
5: -21.3139992, 35.4855690, -21.1363487, 35.4153671, -56.1062164, 55.9895554
6: -42.2133865, 26.0649414, -42.1472054, 25.9738483, -68.1872330, 68.2121429
7: -30.5860023, 34.1704636, -30.3127575, 34.0739594, -64.0184784, 63.8402176
8: -29.1211357, 40.1044807, -28.9452782, 40.0317612, -69.1528931, 69.0497589
9: -24.4278889, 31.6939659, -24.3760891, 31.6238918, -54.9481888, 54.9575119
10: -45.8954849, 31.3971748, -45.8255692, 31.2482967, -77.1437836, 77.2227478
11: -48.9565239, 18.1527615, -48.8549004, 18.0304832, -66.9870071, 67.0076599
12: -52.8040123, 18.5120296, -52.6829300, 18.0868721, -69.2098083, 69.5255737
13: -35.7425346, 38.8211365, -35.6532288, 38.6476707, -74.3902054, 74.4743652
14: -78.4634628, 11.2469635, -78.3267593, 11.0122070, -89.4756699, 89.5737228
15: -30.3083096, 30.2662811, -30.1581764, 30.0614433, -60.3697510, 60.4244576
16: -46.3940811, 30.8143654, -46.1856003, 30.6780910, -77.0650635, 76.9893036
17: -77.9023972, 14.9088707, -77.7914505, 14.6009884, -92.5033875, 92.7003174
18: -45.8229752, 21.3710747, -45.7470398, 21.2804470, -67.1034241, 67.1181183
19: -34.5186119, 11.0014343, -34.4929733, 10.9579000, -45.4765129, 45.4944077
20: -30.6310730, 14.3534374, -30.5599327, 14.2849159, -44.9159889, 44.9133682
21: -42.7248306, 14.9542265, -42.6792412, 14.8899698, -57.6147995, 57.6334686
22: -43.2461472, 17.8151321, -43.1039085, 17.5300217, -60.7761688, 60.9190407
23: -34.4532089, 15.1757717, -34.3910866, 15.1392536, -49.5924606, 49.5668564
24: -36.4826279, 14.9147387, -36.3757744, 14.8755913, -51.3582191, 51.2905121
25: -35.5599785, 17.4111309, -35.4873199, 17.3056259, -52.8656044, 52.8984528
26: -53.5368958, 20.5301800, -53.3751793, 20.1448078, -73.6817017, 73.9053574
27: -36.2980385, 18.9211388, -36.2010193, 18.8895016, -55.1875381, 55.1221581
28: -33.3699341, 19.0436287, -33.3030548, 19.0016251, -52.3715591, 52.3466835
29: -44.9585495, 17.0548058, -44.8397598, 16.8015785, -61.7601280, 61.8945656
30: -42.9761429, 20.0484905, -42.8238983, 19.9751110, -62.9512558, 62.8723907
31: -42.3726196, 15.3443003, -42.3068619, 15.2758427, -57.6484604, 57.6511612
32: -38.5330276, 23.2751274, -38.4588699, 23.1589680, -61.6919937, 61.7339973
33: -48.9350967, 36.0020065, -48.8323021, 35.9279709, -84.8630676, 84.8343048
34: -47.2141800, 21.1399231, -47.1473160, 21.0975380, -68.2769623, 68.2509537
35: -41.7348785, 26.4448185, -41.6730957, 26.4137115, -67.7030945, 67.6662140
36: -42.4433899, 26.7797775, -42.3624268, 26.5851574, -68.1809845, 68.2820663
37: -66.8892059, 22.3526115, -66.8071442, 22.2815208, -86.6757355, 86.6023026
38: -52.6026230, 31.3894844, -52.4893456, 31.2060261, -82.0218277, 82.0576019
39: -60.2931976, 35.4789963, -60.2410278, 35.4154739, -95.7086716, 95.7200241
40: -53.6479950, 28.3619118, -53.5005112, 28.2931614, -81.9411545, 81.8624268
41: -39.1381378, 27.1658154, -39.0726395, 27.0893574, -66.2274933, 66.2384567
42: -32.5587845, 22.0204468, -32.5208206, 21.9302578, -54.4890442, 54.5412674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3858720, upper bound: 37.5075426
time: 61.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4218684, upper bound: 37.5075426
time: 46.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -43.1127815, 35.1434784, -43.0846100, 35.1627121, -78.2754974, 78.2280884
1: -23.3724098, 31.9708195, -23.3359871, 31.9775543, -55.3499641, 55.3068085
2: -18.8990650, 31.9021187, -18.8289909, 31.9075260, -50.7827301, 50.7110329
3: -19.1158237, 35.1508560, -19.0155849, 35.1320801, -53.7926102, 53.7274399
4: -23.4766636, 35.9603806, -23.4853859, 35.9810486, -59.4577103, 59.4457664
5: -21.2840462, 35.4602127, -21.1849327, 35.4620590, -56.1064301, 56.0173416
6: -42.1706963, 26.0214977, -42.1722908, 26.0215549, -68.1922531, 68.1937866
7: -30.5401421, 34.1670151, -30.3988419, 34.1685371, -64.0430450, 63.9248352
8: -29.0625286, 40.0652847, -29.0059128, 40.0820618, -69.1445923, 69.0711975
9: -24.3946571, 31.6588402, -24.3946304, 31.6371632, -54.8849945, 54.9503975
10: -45.8705292, 31.3698769, -45.8571320, 31.2900429, -77.1605682, 77.2270050
11: -48.8832779, 18.0819130, -48.8570328, 18.0897484, -66.9730225, 66.9389496
12: -52.8309631, 18.4851265, -52.8229370, 18.2105389, -69.3659286, 69.5838013
13: -35.7318993, 38.8040085, -35.7291374, 38.6860657, -74.4179688, 74.5331421
14: -78.4608231, 11.2016068, -78.4364624, 11.0768509, -89.5376740, 89.6380692
15: -30.3064880, 30.2457771, -30.2781391, 30.1189747, -60.4254608, 60.5239182
16: -46.3315315, 30.8079128, -46.2605057, 30.8002396, -77.1188660, 77.0574341
17: -77.8554535, 14.7916412, -77.8828278, 14.6933270, -92.5487823, 92.6744690
18: -45.8323746, 21.3287506, -45.7946587, 21.2944336, -67.1268082, 67.1234131
19: -34.4556656, 10.9355431, -34.4821854, 10.9683914, -45.4240570, 45.4177284
20: -30.6003094, 14.3152390, -30.5862236, 14.3042583, -44.9045677, 44.9014626
21: -42.6394730, 14.8662405, -42.6629410, 14.9117651, -57.5512390, 57.5291824
22: -43.2034340, 17.7204895, -43.2257423, 17.6250153, -60.8284492, 60.9462318
23: -34.3783760, 15.0983562, -34.3865547, 15.1540327, -49.5324097, 49.4849091
24: -36.4114227, 14.8684530, -36.3754158, 14.8933191, -51.3047409, 51.2438698
25: -35.4881821, 17.3253231, -35.5089951, 17.3459969, -52.8341789, 52.8343201
26: -53.5536308, 20.4684258, -53.5311699, 20.2594910, -73.8131256, 73.9995956
27: -36.2873611, 18.8963871, -36.2237930, 18.9113560, -55.1987152, 55.1201782
28: -33.3094940, 18.9759560, -33.3051949, 19.0161057, -52.3255997, 52.2811508
29: -44.8678207, 16.9299431, -44.9130058, 16.8835392, -61.7513580, 61.8429489
30: -42.8847122, 19.9806633, -42.8142929, 20.0290108, -62.9137230, 62.7949562
31: -42.3140717, 15.2905645, -42.2965622, 15.3008041, -57.6148758, 57.5871277
32: -38.5149956, 23.2326050, -38.5140839, 23.1749840, -61.6899796, 61.7466888
33: -48.8867493, 35.9983521, -48.8585243, 35.9643478, -84.8510971, 84.8568726
34: -47.1840935, 21.1236897, -47.1677589, 21.1146526, -68.2636642, 68.2562332
35: -41.6940689, 26.4259338, -41.6849594, 26.4225349, -67.6694794, 67.6512299
36: -42.4325104, 26.7647934, -42.4264603, 26.6346188, -68.2045288, 68.3367462
37: -66.8433228, 22.3112831, -66.8443069, 22.2894211, -86.6132507, 86.6302948
38: -52.5679817, 31.3300114, -52.5596771, 31.2312164, -81.9645081, 82.0639038
39: -60.2235336, 35.4502258, -60.2621231, 35.4358826, -95.6594162, 95.7123489
40: -53.5389404, 28.2854996, -53.5491257, 28.3304806, -81.8694229, 81.8346252
41: -39.0917282, 27.1299057, -39.0998917, 27.1227379, -66.2144623, 66.2297974
42: -32.5387650, 22.0184002, -32.5394897, 21.9771233, -54.5158882, 54.5578918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4250087, upper bound: 37.4865876
time: 53.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4647597, upper bound: 37.4865876
time: 57.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -43.1990089, 35.2268715, -43.0962410, 35.2013512, -78.4003601, 78.3231125
1: -23.4252129, 32.0208817, -23.3415546, 31.9999123, -55.4251251, 55.3624344
2: -18.9437084, 31.9421921, -18.8338585, 31.9246712, -50.8481140, 50.7549744
3: -19.1383018, 35.1725655, -19.0189228, 35.1392097, -53.8393707, 53.7519913
4: -23.5436630, 36.0316582, -23.4911613, 36.0122681, -59.5559311, 59.5228195
5: -21.3175774, 35.5160332, -21.1892052, 35.4848709, -56.1773148, 56.0737915
6: -42.2211037, 26.0983219, -42.1788902, 26.0540676, -68.2751694, 68.2772141
7: -30.5943928, 34.2263069, -30.4050980, 34.1939697, -64.1437759, 63.9862900
8: -29.1250229, 40.1421127, -29.0100613, 40.1148071, -69.2398300, 69.1521759
9: -24.4328823, 31.7055855, -24.3994617, 31.6565475, -54.9857178, 54.9920578
10: -45.9068146, 31.4095192, -45.8645554, 31.2993279, -77.2061462, 77.2740784
11: -48.9737854, 18.1716003, -48.8927994, 18.0940456, -67.0678329, 67.0643997
12: -52.8762589, 18.5189648, -52.8375397, 18.2145977, -69.4091797, 69.6807861
13: -35.7754517, 38.8320160, -35.7350426, 38.6939545, -74.4694061, 74.5670624
14: -78.5198288, 11.2522469, -78.4570007, 11.0809059, -89.6007385, 89.7092438
15: -30.3662605, 30.2772999, -30.2971020, 30.1262550, -60.4925156, 60.5744019
16: -46.4085007, 30.8841190, -46.2704315, 30.8322678, -77.2329102, 77.1434631
17: -77.9644623, 14.9186096, -77.9241638, 14.7005959, -92.6650543, 92.8427734
18: -45.8448563, 21.3786716, -45.8019638, 21.3099899, -67.1548462, 67.1806335
19: -34.5278320, 11.0016699, -34.5109673, 10.9715204, -45.4993515, 45.5126381
20: -30.6453972, 14.3565464, -30.6028900, 14.3072491, -44.9526443, 44.9594345
21: -42.7368469, 14.9575672, -42.7012978, 14.9147387, -57.6515846, 57.6588669
22: -43.3266830, 17.8232517, -43.2769890, 17.6289558, -60.9556389, 61.1002426
23: -34.4646683, 15.1808014, -34.4228210, 15.1574011, -49.6220703, 49.6036224
24: -36.4947891, 14.9213467, -36.4105377, 14.8956785, -51.3904686, 51.3318863
25: -35.5897026, 17.4173756, -35.5536652, 17.3513069, -52.9410095, 52.9710388
26: -53.6220856, 20.5376320, -53.5566483, 20.2633591, -73.8854446, 74.0942841
27: -36.3099937, 18.9302654, -36.2327194, 18.9154549, -55.2254486, 55.1629868
28: -33.3790588, 19.0474072, -33.3342552, 19.0194950, -52.3985519, 52.3816605
29: -45.0229645, 17.0582352, -44.9793854, 16.8864975, -61.9094620, 62.0376205
30: -42.9891968, 20.0697517, -42.8568726, 20.0335464, -63.0227432, 62.9266243
31: -42.3813477, 15.3499823, -42.3232040, 15.3046951, -57.6860428, 57.6731873
32: -38.5596771, 23.2829342, -38.5227928, 23.1947060, -61.7543831, 61.8057251
33: -48.9440346, 36.0150833, -48.8673325, 35.9703636, -84.9143982, 84.8824158
34: -47.2201233, 21.1495686, -47.1725769, 21.1204357, -68.3058929, 68.2867050
35: -41.7434387, 26.4522266, -41.6992836, 26.4278297, -67.7224350, 67.7009888
36: -42.4718246, 26.7848396, -42.4323349, 26.6384506, -68.2584381, 68.3539124
37: -66.9060059, 22.3609848, -66.8567581, 22.3106041, -86.7253113, 86.6630249
38: -52.6323814, 31.3978672, -52.5674286, 31.2583046, -82.0934753, 82.1344757
39: -60.3032990, 35.4915390, -60.2718468, 35.4533043, -95.7566071, 95.7633820
40: -53.6584625, 28.4087238, -53.5596123, 28.3882961, -82.0467606, 81.9683380
41: -39.1466751, 27.1888905, -39.1060982, 27.1473198, -66.2939911, 66.2949905
42: -32.5667877, 22.0386524, -32.5458679, 21.9837532, -54.5505409, 54.5845184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=151, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 631

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4250087, upper bound: 37.5075426
time: 59.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4647597, upper bound: 37.5075426
time: 48.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 111.07 seconds
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3950150, upper bound: 37.4893271
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3950150, upper bound: 37.4893271
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3950150, upper bound: 37.5102708
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3950150, upper bound: 37.5102708
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3858720, upper bound: 37.4865876
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3858720, upper bound: 37.4865876
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3858720, upper bound: 37.5075426
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3858720, upper bound: 37.5075426
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3858720, upper bound: 37.4865876
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.4218684, upper bound: 37.4865876
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.3858720, upper bound: 37.5075426
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.4218684, upper bound: 37.5075426
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.4250087, upper bound: 37.4865876
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.4647597, upper bound: 37.4865876
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.4250087, upper bound: 37.5075426
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 111.07
Output dim: 8, lower bound: -37.4647597, upper bound: 37.5075426

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -43.0545044, 35.0647812, -42.8433533, 34.9510460, -78.0055542, 77.9081345
1: -23.3425655, 31.8968086, -23.2039471, 31.7755318, -55.1180954, 55.1007538
2: -18.8675518, 31.7908058, -18.6343040, 31.6456451, -50.4916000, 50.4047508
3: -19.0844307, 35.0044746, -18.7733536, 34.7806473, -53.4243622, 53.3232498
4: -23.4365883, 35.8967094, -23.3281651, 35.7985306, -59.2351189, 59.2248764
5: -21.2516518, 35.3245583, -20.9286041, 35.1240501, -55.7450409, 55.6159554
6: -42.1278381, 25.9080334, -42.0170097, 25.7344418, -67.8622818, 67.9250412
7: -30.4943619, 33.9801598, -30.1183796, 33.7218399, -63.5605469, 63.4490051
8: -29.0354691, 39.9404564, -28.8010559, 39.7694778, -68.8049469, 68.7415161
9: -24.3091850, 31.5901375, -24.1545582, 31.3569260, -54.4874039, 54.6572266
10: -45.7217255, 31.3102379, -45.4901047, 30.9316978, -76.6534271, 76.8003387
11: -48.7969475, 18.0090179, -48.6817169, 17.8213997, -66.6183472, 66.6907349
12: -52.4187775, 18.4317570, -51.8778839, 17.6251183, -68.3495941, 68.6051636
13: -35.5937462, 38.7537460, -35.3596077, 38.4458046, -74.0395508, 74.1133575
14: -78.1297455, 11.1744041, -77.6335526, 10.7152901, -88.8450317, 88.8079529
15: -30.0799732, 30.1977711, -29.6997547, 29.8623142, -59.9422874, 59.8975258
16: -46.2441483, 30.5775948, -46.0161324, 30.2445755, -76.4802704, 76.5841980
17: -77.4991684, 14.7456264, -77.0521393, 14.2290535, -91.7282257, 91.7977676
18: -45.6891098, 21.2862854, -45.4283752, 21.1042213, -66.7933350, 66.7146606
19: -34.3976898, 10.9215670, -34.3577347, 10.8964710, -45.2941589, 45.2793007
20: -30.5092506, 14.2933235, -30.3229752, 14.1924763, -44.7017288, 44.6162987
21: -42.5697784, 14.8439322, -42.5069695, 14.7927885, -57.3625679, 57.3509026
22: -42.8734055, 17.6840210, -42.4732895, 17.2776794, -60.1510849, 60.1573105
23: -34.3147812, 15.0626898, -34.1998520, 15.0470362, -49.3618164, 49.2625427
24: -36.3568420, 14.8122120, -36.1841660, 14.7413845, -51.0982285, 50.9963760
25: -35.3849487, 17.2935181, -35.2417297, 17.1866646, -52.5716133, 52.5352478
26: -53.1431427, 20.4230118, -52.5769196, 19.7772350, -72.9203796, 72.9999313
27: -36.2338181, 18.8495388, -36.0233078, 18.7784729, -55.0122910, 54.8728485
28: -33.2608414, 18.9321365, -33.1234474, 18.8864899, -52.1473312, 52.0555840
29: -44.6162033, 16.9076633, -44.2996521, 16.5587730, -61.1749763, 61.2073135
30: -42.8248749, 19.7980576, -42.5322723, 19.5702076, -62.3950806, 62.3303299
31: -42.2587967, 15.2675381, -42.1914635, 15.1947680, -57.4535637, 57.4589996
32: -38.3872604, 23.1849632, -38.1787109, 22.9784279, -61.3656883, 61.3636742
33: -48.8206940, 35.8741608, -48.5651550, 35.6231918, -84.4438858, 84.4393158
34: -47.1401062, 21.0117416, -46.9462051, 20.8326645, -67.9373093, 67.9159546
35: -41.6406288, 26.3169727, -41.4438744, 26.1591969, -67.3573608, 67.2664795
36: -42.3224258, 26.7312336, -42.1156769, 26.4839897, -67.9513397, 67.9868164
37: -66.7597275, 22.2532177, -66.5869293, 22.1071186, -86.3211823, 86.2602463
38: -52.4097748, 31.2639542, -52.1130409, 30.9822006, -81.5663605, 81.5410461
39: -60.1473083, 35.3384399, -60.0055237, 35.1174164, -95.2647247, 95.3439636
40: -53.4825516, 28.0918980, -53.2598686, 27.8863087, -81.3688583, 81.3517685
41: -39.0470314, 27.0439529, -38.9451408, 26.8964252, -65.9434586, 65.9890900
42: -32.4761848, 21.9449997, -32.3548012, 21.7579212, -54.2341080, 54.2998009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3915621, upper bound: 37.4460993
time: 49.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3915621, upper bound: 37.4844381
time: 51.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -43.0802994, 35.0959015, -42.8906021, 35.0196762, -78.0999756, 77.9865036
1: -23.3568840, 31.9289780, -23.2160645, 31.8471260, -55.2040100, 55.1450424
2: -18.8804398, 31.8291245, -18.6627884, 31.7296677, -50.5857010, 50.4706230
3: -19.0987167, 35.0598831, -18.8185387, 34.9032440, -53.5358887, 53.4281921
4: -23.4522419, 35.9310303, -23.3704338, 35.8784637, -59.3209991, 59.3014641
5: -21.2661591, 35.3644066, -20.9684067, 35.2120399, -55.8352661, 55.6968689
6: -42.1523132, 25.9421635, -42.0683823, 25.8069000, -67.9592133, 68.0105438
7: -30.5131531, 34.0474701, -30.1352978, 33.8607140, -63.7080231, 63.5343857
8: -29.0488262, 39.9925537, -28.8384113, 39.8850517, -68.9338760, 68.8309631
9: -24.3288021, 31.6378632, -24.2120552, 31.4672260, -54.5768547, 54.7507172
10: -45.7462349, 31.3391056, -45.5486908, 30.9965878, -76.7428207, 76.8877945
11: -48.8416061, 18.0555096, -48.7460403, 17.9197273, -66.7613373, 66.8015518
12: -52.5946312, 18.4583416, -52.2876434, 17.7593117, -68.6676559, 68.9968567
13: -35.6499176, 38.7782784, -35.4762344, 38.5139389, -74.1638565, 74.2545166
14: -78.2691269, 11.1894531, -77.9608002, 10.7962189, -89.0653458, 89.1502533
15: -30.1780853, 30.2225609, -29.9212952, 29.8909111, -60.0689964, 60.1438560
16: -46.2864380, 30.7216988, -46.1148376, 30.5440178, -76.8139191, 76.8263092
17: -77.6881638, 14.7691517, -77.5037918, 14.3556824, -92.0438461, 92.2729416
18: -45.7719040, 21.3070107, -45.6205559, 21.1598969, -66.9318008, 66.9275665
19: -34.4260368, 10.9301491, -34.3954430, 10.9131746, -45.3392105, 45.3255920
20: -30.5611439, 14.3046513, -30.4476395, 14.2315187, -44.7926636, 44.7522888
21: -42.6052551, 14.8565960, -42.5673866, 14.8204937, -57.4257507, 57.4239807
22: -43.0342445, 17.7017765, -42.8176460, 17.3490086, -60.3832550, 60.5194244
23: -34.3475723, 15.0738668, -34.2718048, 15.0665474, -49.4141197, 49.3456726
24: -36.3843689, 14.8431091, -36.2537193, 14.8114662, -51.1958351, 51.0968285
25: -35.4368134, 17.3066216, -35.3592644, 17.2259064, -52.6627197, 52.6658859
26: -53.3264236, 20.4422054, -53.0011139, 19.8711720, -73.1975937, 73.4433212
27: -36.2613525, 18.8670750, -36.0754776, 18.8091469, -55.0704994, 54.9425507
28: -33.2856369, 18.9423389, -33.1791763, 18.9080849, -52.1937218, 52.1215134
29: -44.7421989, 16.9175663, -44.5972099, 16.6300278, -61.3722267, 61.5147781
30: -42.8545990, 19.8928452, -42.6222534, 19.7820511, -62.6366501, 62.5150986
31: -42.2864037, 15.2804861, -42.2354050, 15.2190342, -57.5054398, 57.5158920
32: -38.4533081, 23.2090111, -38.3294525, 23.0523243, -61.5056305, 61.5384636
33: -48.8558617, 35.9611320, -48.6752243, 35.8291512, -84.6850128, 84.6363525
34: -47.1643906, 21.0615635, -47.0215073, 20.9496593, -68.0790482, 68.0473785
35: -41.6692123, 26.3679638, -41.5227890, 26.2810707, -67.5089111, 67.4349976
36: -42.3877945, 26.7452240, -42.2627525, 26.5411453, -68.0749207, 68.1277542
37: -66.7983246, 22.2876282, -66.6707916, 22.1852608, -86.4532776, 86.4780807
38: -52.5254440, 31.2933254, -52.3821068, 31.0885086, -81.7933426, 81.7921219
39: -60.1777763, 35.4179459, -60.0895500, 35.3041916, -95.4819641, 95.5074921
40: -53.5087128, 28.1978531, -53.3425064, 28.1280804, -81.6367950, 81.5403595
41: -39.0713730, 27.0760040, -38.9997635, 26.9618225, -66.0331955, 66.0757675
42: -32.5197754, 21.9734688, -32.4607735, 21.8195858, -54.3393631, 54.4342422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4182988, upper bound: 37.4460993
time: 55.02 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4182988, upper bound: 37.4844381
time: 51.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -43.1407433, 35.1481323, -42.8549881, 34.9897118, -78.1304550, 78.0031204
1: -23.3953857, 31.9468803, -23.2095280, 31.7978210, -55.1932068, 55.1564102
2: -18.9122028, 31.8308601, -18.6391945, 31.6627922, -50.5570145, 50.4487152
3: -19.1069145, 35.0262032, -18.7767029, 34.7877274, -53.4710693, 53.3477783
4: -23.5036297, 35.9679718, -23.3339081, 35.8297539, -59.3333817, 59.3018799
5: -21.2851906, 35.3803711, -20.9329033, 35.1468315, -55.8159027, 55.6724930
6: -42.1782761, 25.9848442, -42.0236206, 25.7668343, -67.9451141, 68.0084686
7: -30.5486336, 34.0393982, -30.1246681, 33.7472572, -63.6613693, 63.5105286
8: -29.0979729, 40.0172958, -28.8052101, 39.8021774, -68.9001465, 68.8225098
9: -24.3474197, 31.6368103, -24.1594124, 31.3763084, -54.5879440, 54.6988487
10: -45.7580490, 31.3498650, -45.4975052, 30.9410439, -76.6990967, 76.8473663
11: -48.8873749, 18.0988884, -48.7175522, 17.8256226, -66.7129974, 66.8164368
12: -52.4640923, 18.4656048, -51.8924217, 17.6291866, -68.3928375, 68.7019882
13: -35.6374283, 38.7817383, -35.3654861, 38.4537315, -74.0911560, 74.1472244
14: -78.1887970, 11.2251205, -77.6540680, 10.7193336, -88.9081268, 88.8791885
15: -30.1395893, 30.2293396, -29.7186852, 29.8696480, -60.0092392, 59.9480247
16: -46.3211250, 30.6537838, -46.0261002, 30.2765656, -76.5942993, 76.6702881
17: -77.6081543, 14.8725662, -77.0934143, 14.2363548, -91.8445129, 91.9659805
18: -45.7015953, 21.3359070, -45.4356537, 21.1196938, -66.8212891, 66.7715607
19: -34.4698334, 10.9876909, -34.3865433, 10.8995819, -45.3694153, 45.3742332
20: -30.5543308, 14.3346128, -30.3395748, 14.1954517, -44.7497826, 44.6741867
21: -42.6670914, 14.9352036, -42.5453568, 14.7957401, -57.4628296, 57.4805603
22: -42.9966431, 17.7868233, -42.5244980, 17.2816277, -60.2782707, 60.3113213
23: -34.4010315, 15.1451464, -34.2361221, 15.0503960, -49.4514275, 49.3812675
24: -36.4401436, 14.8651295, -36.2193069, 14.7437172, -51.1838608, 51.0844345
25: -35.4864883, 17.3855896, -35.2863464, 17.1920204, -52.6785088, 52.6719360
26: -53.2116089, 20.4922504, -52.6023674, 19.7810612, -72.9926682, 73.0946198
27: -36.2564163, 18.8834305, -36.0322876, 18.7824707, -55.0388870, 54.9157181
28: -33.3304062, 19.0036240, -33.1525612, 18.8898792, -52.2202835, 52.1561852
29: -44.7714005, 17.0359726, -44.3660049, 16.5617294, -61.3331299, 61.4019775
30: -42.9293594, 19.8872871, -42.5748367, 19.5744991, -62.5038605, 62.4621239
31: -42.3259659, 15.3269730, -42.2181511, 15.1986084, -57.5245743, 57.5451241
32: -38.4319191, 23.2352848, -38.1873398, 22.9982033, -61.4301224, 61.4226227
33: -48.8779716, 35.8908768, -48.5739975, 35.6291618, -84.5071335, 84.4648743
34: -47.1761093, 21.0374851, -46.9510155, 20.8383255, -67.9794464, 67.9464188
35: -41.6899948, 26.3432732, -41.4582024, 26.1644936, -67.4103241, 67.3163757
36: -42.3619843, 26.7512093, -42.1215286, 26.4878216, -68.0054169, 68.0040436
37: -66.8224258, 22.3027687, -66.5993805, 22.1282349, -86.4333496, 86.2929230
38: -52.4742050, 31.3317699, -52.1207962, 31.0093250, -81.6955414, 81.6116180
39: -60.2271309, 35.3796806, -60.0152855, 35.1348419, -95.3619690, 95.3949661
40: -53.6021347, 28.2150440, -53.2703819, 27.9441128, -81.5462494, 81.4854279
41: -39.1020279, 27.1029320, -38.9513969, 26.9209614, -66.0229874, 66.0543289
42: -32.5041199, 21.9652386, -32.3611031, 21.7645035, -54.2686234, 54.3263397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3915621, upper bound: 37.4671010
time: 69.02 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3915621, upper bound: 37.5053911
time: 64.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -43.1665001, 35.1792908, -42.9022293, 35.0583038, -78.2248077, 78.0815201
1: -23.4097099, 31.9790840, -23.2216091, 31.8694782, -55.2791901, 55.2006912
2: -18.9250698, 31.8691807, -18.6676769, 31.7468128, -50.6510925, 50.5145950
3: -19.1211853, 35.0815926, -18.8218727, 34.9103546, -53.5826111, 53.4527168
4: -23.5192566, 36.0022888, -23.3762188, 35.9096947, -59.4225616, 59.3785095
5: -21.2996807, 35.4202080, -20.9727020, 35.2348633, -55.9061966, 55.7534103
6: -42.2027359, 26.0190029, -42.0749817, 25.8394604, -68.0421982, 68.0939865
7: -30.5674553, 34.1067734, -30.1415749, 33.8861427, -63.8088531, 63.5959015
8: -29.1113014, 40.0693817, -28.8425694, 39.9177933, -69.0290985, 68.9119492
9: -24.3670235, 31.6845493, -24.2168808, 31.4866524, -54.6775589, 54.7923775
10: -45.7824860, 31.3787346, -45.5560913, 31.0059166, -76.7884064, 76.9348297
11: -48.9320602, 18.1453056, -48.7818489, 17.9239273, -66.8559875, 66.9271545
12: -52.6399307, 18.4921856, -52.3022270, 17.7633648, -68.7109222, 69.0937347
13: -35.6935883, 38.8062668, -35.4821129, 38.5218506, -74.2154388, 74.2883759
14: -78.3281708, 11.2401600, -77.9813309, 10.8002930, -89.1284637, 89.2214890
15: -30.2376862, 30.2541275, -29.9402447, 29.8982086, -60.1358948, 60.1943741
16: -46.3634262, 30.7979050, -46.1247444, 30.5760860, -76.9280090, 76.9123688
17: -77.7971497, 14.8960896, -77.5451431, 14.3629627, -92.1601105, 92.4412308
18: -45.7844086, 21.3567657, -45.6278610, 21.1754684, -66.9598770, 66.9846268
19: -34.4981956, 10.9962692, -34.4242210, 10.9163017, -45.4144974, 45.4204903
20: -30.6062393, 14.3459482, -30.4642620, 14.2344942, -44.8407326, 44.8102112
21: -42.7025909, 14.9479113, -42.6057510, 14.8234577, -57.5260468, 57.5536613
22: -43.1574860, 17.8045406, -42.8688812, 17.3529072, -60.5103912, 60.6734238
23: -34.4338150, 15.1563196, -34.3080978, 15.0699053, -49.5037193, 49.4644165
24: -36.4677353, 14.8960238, -36.2888412, 14.8138161, -51.2815514, 51.1848640
25: -35.5383377, 17.3986721, -35.4039116, 17.2312393, -52.7695770, 52.8025818
26: -53.3948975, 20.5114040, -53.0265694, 19.8750458, -73.2699432, 73.5379715
27: -36.2839241, 18.9009762, -36.0844612, 18.8131886, -55.0971146, 54.9854355
28: -33.3551941, 19.0138245, -33.2082634, 18.9114761, -52.2666702, 52.2220879
29: -44.8974113, 17.0458603, -44.6636162, 16.6329498, -61.5303612, 61.7094765
30: -42.9590721, 19.9819946, -42.6648560, 19.7863693, -62.7454414, 62.6468506
31: -42.3536415, 15.3399286, -42.2620506, 15.2228823, -57.5765228, 57.6019783
32: -38.4979744, 23.2593422, -38.3380775, 23.0720749, -61.5700493, 61.5974197
33: -48.9131012, 35.9778366, -48.6840248, 35.8351707, -84.7482758, 84.6618652
34: -47.2003860, 21.0873566, -47.0262718, 20.9554577, -68.1212540, 68.0778885
35: -41.7185783, 26.3942986, -41.5371552, 26.2863789, -67.5618439, 67.4848404
36: -42.4272385, 26.7651730, -42.2686234, 26.5450478, -68.1289368, 68.1449738
37: -66.8610077, 22.3372192, -66.6833191, 22.2064114, -86.5653076, 86.5107727
38: -52.5898361, 31.3611183, -52.3898506, 31.1156292, -81.9224319, 81.8626404
39: -60.2575531, 35.4592590, -60.0992317, 35.3216019, -95.5791550, 95.5584869
40: -53.6282578, 28.3209858, -53.3529892, 28.1858768, -81.8141327, 81.6739731
41: -39.1263657, 27.1349678, -39.0060196, 26.9863701, -66.1127319, 66.1409912
42: -32.5477371, 21.9937019, -32.4671326, 21.8262157, -54.3739548, 54.4608345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 631

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4182988, upper bound: 37.4671010
time: 56.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4182988, upper bound: 37.5053911
time: 63.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.1485176, 35.1224251, -42.8765602, 34.9343987, -78.0829163, 77.9989853
1: -23.3983822, 31.9563770, -23.2142181, 31.8225937, -55.2209778, 55.1705933
2: -18.9103413, 31.8384209, -18.6503830, 31.6810856, -50.5712891, 50.4640312
3: -19.1110344, 35.0810890, -18.8339272, 34.9110336, -53.5884857, 53.4490242
4: -23.4975739, 35.9414978, -23.3347206, 35.7758636, -59.2703934, 59.2650566
5: -21.2862072, 35.3731308, -20.9597054, 35.1294098, -55.7932968, 55.6858482
6: -42.1953163, 25.9973259, -42.0611191, 25.7980022, -67.9933167, 68.0584412
7: -30.5552006, 34.0873566, -30.1482735, 33.8491249, -63.7597809, 63.5684509
8: -29.1005573, 40.0407562, -28.8459206, 39.8518448, -68.9524002, 68.8866730
9: -24.3486099, 31.6698532, -24.1824532, 31.4625454, -54.6765137, 54.7518730
10: -45.6983337, 31.3635731, -45.3746109, 30.9442940, -76.6426239, 76.7381821
11: -48.8806915, 18.1483955, -48.6685867, 17.9366798, -66.8173676, 66.8169861
12: -52.6031952, 18.4786167, -52.2061272, 17.7902393, -68.6670151, 69.0106049
13: -35.7032547, 38.7884674, -35.5139160, 38.5016289, -74.2048798, 74.3023834
14: -78.2481232, 11.2313271, -77.8013153, 10.7670956, -89.0152206, 89.0326385
15: -30.2234287, 30.2499065, -29.9247074, 29.9080276, -60.1314545, 60.1746140
16: -46.3540192, 30.7932720, -46.1335907, 30.5707417, -76.9161224, 76.9138794
17: -77.7185211, 14.8889427, -77.3562927, 14.3444176, -92.0629425, 92.2452393
18: -45.7756996, 21.3391762, -45.6605835, 21.1470242, -66.9227219, 66.9997559
19: -34.4681587, 10.9896240, -34.3485031, 10.9031668, -45.3713264, 45.3381271
20: -30.5803947, 14.3377819, -30.4139557, 14.2121944, -44.7925873, 44.7517395
21: -42.6566467, 14.9401112, -42.4914589, 14.7999277, -57.4565735, 57.4315720
22: -43.1170158, 17.7992630, -42.7701263, 17.3621044, -60.4791183, 60.5693893
23: -34.3786392, 15.1502333, -34.1873169, 15.0407066, -49.4193459, 49.3375511
24: -36.4384003, 14.8937950, -36.2313194, 14.8002453, -51.2386475, 51.1251144
25: -35.4910965, 17.3851166, -35.3022156, 17.1959286, -52.6870270, 52.6873322
26: -53.3748360, 20.5055790, -52.9694214, 19.9081173, -73.2829514, 73.4749985
27: -36.2749138, 18.8919773, -36.0782890, 18.8044128, -55.0793266, 54.9702682
28: -33.3071976, 19.0037804, -33.1048584, 18.9011154, -52.2083130, 52.1086388
29: -44.8251343, 17.0375061, -44.4931221, 16.6236477, -61.4487839, 61.5306282
30: -42.9048615, 19.9829292, -42.5661545, 19.7961235, -62.7009850, 62.5490837
31: -42.3245621, 15.3356314, -42.1899071, 15.2199640, -57.5445251, 57.5255394
32: -38.4962540, 23.2543430, -38.3383865, 23.0910912, -61.5873451, 61.5927277
33: -48.8982315, 35.9470367, -48.6415939, 35.7780609, -84.6762924, 84.5886307
34: -47.1807175, 21.0776634, -46.9840012, 20.9547043, -68.1015091, 68.0229797
35: -41.7096443, 26.3859825, -41.5144234, 26.2657032, -67.5242538, 67.4251633
36: -42.4374809, 26.7535305, -42.2889328, 26.5568161, -68.1339111, 68.1495209
37: -66.8473511, 22.2339153, -66.6729584, 22.0008774, -86.3063507, 86.2873154
38: -52.5959167, 31.2961102, -52.4170265, 31.0095329, -81.7891846, 81.8322754
39: -60.2388229, 35.3897552, -60.0523376, 35.1810532, -95.4198761, 95.4420929
40: -53.6189537, 28.2423496, -53.3242836, 28.0057182, -81.6246719, 81.5666351
41: -39.1154251, 27.0942211, -38.9731789, 26.9069672, -66.0223923, 66.0673981
42: -32.5203629, 21.9815960, -32.4189873, 21.8487759, -54.3691406, 54.4005814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3822942, upper bound: 37.4750764
time: 64.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3822942, upper bound: 37.5053911
time: 52.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.1744385, 35.2080574, -42.9707069, 35.1270294, -78.3014679, 78.1787643
1: -23.4135017, 32.0059776, -23.2716122, 31.9327888, -55.3462906, 55.2775879
2: -18.9293842, 31.8967667, -18.7224026, 31.8124485, -50.7227936, 50.5980225
3: -19.1250610, 35.1091232, -18.8720741, 34.9775429, -53.6743469, 53.5382042
4: -23.5224762, 36.0182266, -23.4098778, 35.9475861, -59.4700394, 59.4281044
5: -21.3027859, 35.4504814, -21.0251350, 35.3078995, -55.9910049, 55.8412857
6: -42.2102928, 26.0521641, -42.1083603, 25.9233017, -68.1335907, 68.1605225
7: -30.5753860, 34.1634369, -30.2335854, 34.0251541, -63.9610291, 63.7480316
8: -29.1146984, 40.1069107, -28.9071617, 40.0047226, -69.1194229, 69.0140686
9: -24.3711319, 31.6957855, -24.2374382, 31.5194397, -54.7430840, 54.8180122
10: -45.7932129, 31.3903084, -45.5951805, 31.0559940, -76.8492050, 76.9854889
11: -48.9477615, 18.1651707, -48.8152618, 18.0036945, -66.9514542, 66.9804306
12: -52.7117577, 18.4984169, -52.4574852, 17.8900890, -68.9220810, 69.2966919
13: -35.7267380, 38.8163338, -35.5759659, 38.5649185, -74.2916565, 74.3923035
14: -78.3836823, 11.2450790, -78.1132736, 10.8683300, -89.2520142, 89.3583527
15: -30.2959728, 30.2649956, -30.0950222, 29.9620247, -60.2579956, 60.3600159
16: -46.3775444, 30.8677330, -46.2109108, 30.7434387, -77.1129608, 77.0691071
17: -77.8583450, 14.9055271, -77.6773376, 14.4643917, -92.3227386, 92.5828629
18: -45.8061485, 21.3613205, -45.6846390, 21.1920853, -66.9982300, 67.0459595
19: -34.5070724, 10.9965620, -34.4423218, 10.9345932, -45.4416656, 45.4388847
20: -30.6202507, 14.3488140, -30.5083923, 14.2569780, -44.8772278, 44.8572083
21: -42.7141838, 14.9515886, -42.6278000, 14.8563986, -57.5705833, 57.5793877
22: -43.2376022, 17.8122787, -43.0462646, 17.4516106, -60.6892128, 60.8585434
23: -34.4447479, 15.1613750, -34.3385239, 15.0926857, -49.5374336, 49.4999008
24: -36.4793854, 14.9025936, -36.3232117, 14.8364182, -51.3158035, 51.2258072
25: -35.5679283, 17.4045906, -35.4754639, 17.2771282, -52.8450546, 52.8800545
26: -53.4794235, 20.5183697, -53.2103729, 19.9932137, -73.4726410, 73.7287445
27: -36.2955780, 18.9090881, -36.1169891, 18.8432446, -55.1388245, 55.0260773
28: -33.3630447, 19.0174408, -33.2344360, 18.9318314, -52.2948761, 52.2518768
29: -44.9609642, 17.0488567, -44.8036880, 16.7172451, -61.6782074, 61.8525467
30: -42.9714661, 20.0032024, -42.6976967, 19.8506699, -62.8221359, 62.7008972
31: -42.3618622, 15.3457737, -42.2774200, 15.2581463, -57.6200104, 57.6231918
32: -38.5235062, 23.2670345, -38.4010620, 23.1092663, -61.6327744, 61.6680984
33: -48.9217453, 35.9906235, -48.7206802, 35.8800201, -84.8017654, 84.7113037
34: -47.2048645, 21.0967979, -47.0453720, 20.9797592, -68.1498489, 68.1034241
35: -41.7269669, 26.4015808, -41.5643959, 26.3015881, -67.5744476, 67.4974060
36: -42.4560585, 26.7696018, -42.3480225, 26.5961647, -68.1943893, 68.2447891
37: -66.8774872, 22.3445816, -66.7341766, 22.2342339, -86.5850220, 86.5153046
38: -52.6192780, 31.3675919, -52.4700890, 31.1610165, -81.9671249, 81.9898453
39: -60.2673340, 35.4709396, -60.1316948, 35.3581619, -95.6254959, 95.6026306
40: -53.6384354, 28.3667488, -53.4139175, 28.2843933, -81.9228287, 81.7806702
41: -39.1346169, 27.1581860, -39.0398865, 27.0521889, -66.1868057, 66.1980743
42: -32.5541344, 22.0114975, -32.4878502, 21.8819389, -54.4360733, 54.4993477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=400, inp2_unstable=400, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.3822942, upper bound: 37.4750764
time: 54.25 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3822942, upper bound: 37.5053911
time: 664.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -43.1637306, 35.1114349, -42.9288712, 34.9414253, -78.1051559, 78.0403061
1: -23.4053421, 31.9436455, -23.2303925, 31.8290005, -55.2343445, 55.1740379
2: -18.9192467, 31.8554878, -18.7036057, 31.7295990, -50.6279068, 50.5354500
3: -19.1193638, 35.1164322, -18.9274216, 35.0075531, -53.6860199, 53.5888062
4: -23.5142460, 35.9366455, -23.3794594, 35.7945099, -59.3087540, 59.3092499
5: -21.2967987, 35.4075050, -21.0679474, 35.2337494, -55.9042206, 55.8346634
6: -42.1978493, 26.0094662, -42.0971642, 25.8460140, -68.0438614, 68.1066284
7: -30.5651779, 34.0936203, -30.2244339, 33.8949852, -63.8131256, 63.6624222
8: -29.1062889, 40.0377350, -28.8805084, 39.8761673, -68.9824524, 68.9182434
9: -24.4043064, 31.6670914, -24.3155460, 31.5628548, -54.8743057, 54.8786774
10: -45.7997055, 31.3692856, -45.6012230, 31.1319599, -76.9316635, 76.9705048
11: -48.8874664, 18.1353912, -48.6985092, 17.9604683, -66.8479309, 66.8339005
12: -52.6944618, 18.4912624, -52.4273567, 17.9828320, -68.9629364, 69.2325134
13: -35.7179832, 38.7921677, -35.5870895, 38.5789261, -74.2969055, 74.3792572
14: -78.3265839, 11.2327080, -78.0088806, 10.9087563, -89.2353363, 89.2415924
15: -30.2345409, 30.2507744, -29.9870110, 30.0054741, -60.2400131, 60.2377853
16: -46.3698997, 30.7383881, -46.1049194, 30.4991875, -76.8615570, 76.8303833
17: -77.7612228, 14.8916798, -77.4645081, 14.4781532, -92.2393799, 92.3561859
18: -45.7918549, 21.3455505, -45.7201691, 21.2194633, -67.0113220, 67.0657196
19: -34.4791145, 10.9941549, -34.3967209, 10.9244785, -45.4035950, 45.3908768
20: -30.5906639, 14.3420639, -30.4631271, 14.2386045, -44.8292694, 44.8051910
21: -42.6665459, 14.9423580, -42.5396271, 14.8313980, -57.4979439, 57.4819870
22: -43.1243591, 17.8015213, -42.8225250, 17.4374466, -60.5618057, 60.6240463
23: -34.3863678, 15.1642151, -34.2368240, 15.0848207, -49.4711876, 49.4010391
24: -36.4408760, 14.9056520, -36.2805023, 14.8379707, -51.2788467, 51.1861534
25: -35.4822884, 17.3911934, -35.3102646, 17.2222900, -52.7045784, 52.7014580
26: -53.4310570, 20.5166073, -53.1289749, 20.0565186, -73.4875793, 73.6455841
27: -36.2767258, 18.9029961, -36.1600189, 18.8454590, -55.1221848, 55.0630150
28: -33.3126068, 19.0295105, -33.1666794, 18.9676380, -52.2802429, 52.1961899
29: -44.8213921, 17.0428543, -44.5232811, 16.7047729, -61.5261650, 61.5661354
30: -42.9084625, 20.0275574, -42.6853333, 19.9143143, -62.8227768, 62.7128906
31: -42.3345642, 15.3338909, -42.2159309, 15.2359104, -57.5704727, 57.5498199
32: -38.5044327, 23.2619743, -38.3922043, 23.1388779, -61.6433105, 61.6541786
33: -48.9109535, 35.9578323, -48.7504616, 35.8232574, -84.7342072, 84.7082977
34: -47.1883430, 21.1203804, -47.0783653, 21.0700550, -68.2241364, 68.1622925
35: -41.7170715, 26.4288177, -41.6216240, 26.3758202, -67.6420975, 67.5919647
36: -42.4243660, 26.7628803, -42.2972336, 26.5414867, -68.1055832, 68.1840820
37: -66.8583984, 22.2409134, -66.7417908, 22.0419540, -86.3607559, 86.3694000
38: -52.5784454, 31.3158646, -52.4282570, 31.0412731, -81.8097839, 81.8919449
39: -60.2640266, 35.3965759, -60.1576729, 35.2328644, -95.4968872, 95.5542450
40: -53.6278610, 28.2362080, -53.4077263, 28.0092640, -81.6371231, 81.6439362
41: -39.1183701, 27.1010971, -39.0033417, 26.9408932, -66.0592651, 66.1044388
42: -32.5230522, 21.9898930, -32.4431496, 21.8941860, -54.4172363, 54.4330444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=150, inp2_unstable=150, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=399, inp2_unstable=399, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 748

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 665

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -37.4284091, upper bound: 37.4750764
time: 53.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.3822942, upper bound: 37.5053911
time: 57.91 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 113.68 seconds
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3915621, upper bound: 37.4460993
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3915621, upper bound: 37.4844381
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.4182988, upper bound: 37.4460993
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.4182988, upper bound: 37.4844381
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3915621, upper bound: 37.4671010
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3915621, upper bound: 37.5053911
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.4182988, upper bound: 37.4671010
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.4182988, upper bound: 37.5053911
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3822942, upper bound: 37.4750764
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3822942, upper bound: 37.5053911
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3822942, upper bound: 37.4750764
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3822942, upper bound: 37.5053911
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.4284091, upper bound: 37.4750764
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 113.68
Output dim: 8, lower bound: -37.3822942, upper bound: 37.5053911
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 113.68
Output dim: 8, lower bound: -37.4218684, upper bound: 37.5075426
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 113.68
Output dim: 8, lower bound: -37.4250087, upper bound: 37.5075426
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 113.68
Output dim: 8, lower bound: -37.4647597, upper bound: 37.5075426

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 80.89 + 3625.87 = 3706.76 seconds
