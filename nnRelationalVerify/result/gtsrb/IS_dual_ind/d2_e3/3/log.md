## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.21 + 80.39 = 82.60 seconds
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 664

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5368227, upper bound: 37.4945694
time: 54.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5368227, upper bound: 37.5368226
time: 43.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 97.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 97.67
Output dim: 8, lower bound: -37.5368227, upper bound: 37.4945694
IS_A2, status: Status.UNKNOWN, split count: 1, time: 97.67
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

Time for backsubstitution: 1.79 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
time: 49.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
time: 200.20 seconds

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

Time for backsubstitution: 1.80 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836
time: 48.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836
time: 59.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 109.82 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 109.82
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 109.82
Output dim: 8, lower bound: -37.4840359, upper bound: 37.4868853
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 109.82
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 109.82
Output dim: 8, lower bound: -37.4840359, upper bound: 37.5284836

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -43.0678711, 35.1959381, -42.9775238, 35.1464386, -78.2143097, 78.1734619
1: -23.3297520, 31.9901371, -23.2765121, 31.9497414, -55.2794952, 55.2666473
2: -18.8185635, 31.8422127, -18.7269135, 31.8021984, -50.6007080, 50.5486259
3: -19.0067139, 35.0466576, -18.8789692, 34.9719772, -53.5428696, 53.4823036
4: -23.4753475, 36.0002403, -23.4168816, 35.9612923, -59.4366379, 59.4171219
5: -21.1728287, 35.3810730, -21.0289879, 35.3001328, -55.8481140, 55.7782135
6: -42.1655579, 26.0266285, -42.1138153, 25.9501114, -68.1156693, 68.1404419
7: -30.3832035, 34.0991249, -30.2381992, 34.0302734, -63.7716293, 63.6888199
8: -29.0022583, 40.0438461, -28.9136162, 39.9974213, -68.9996796, 68.9574585
9: -24.3396263, 31.6453476, -24.2485027, 31.5272026, -54.7034378, 54.7578812
10: -45.7359848, 31.2713909, -45.5995445, 31.0612221, -76.7972107, 76.8709335
11: -48.8626175, 18.1503220, -48.8313103, 18.0707321, -66.9333496, 66.9816284
12: -52.5324059, 18.1850891, -52.4021950, 17.8920269, -68.7500610, 68.9155807
13: -35.6986961, 38.6724014, -35.6120110, 38.5710373, -74.2697296, 74.2844086
14: -78.2359695, 11.0706139, -78.0874634, 10.8709087, -89.1068802, 89.1580811
15: -30.2249241, 30.1117477, -30.1319313, 29.9680939, -60.1930161, 60.2436790
16: -46.2336273, 30.8550911, -46.2191353, 30.8000374, -77.0259247, 77.0673981
17: -77.7272186, 14.6894588, -77.6460876, 14.4736633, -92.2008820, 92.3355484
18: -45.7322311, 21.3081646, -45.6760559, 21.2247696, -66.9570007, 66.9842224
19: -34.4824181, 10.9878206, -34.4454536, 10.9587612, -45.4411774, 45.4332733
20: -30.5608082, 14.2994556, -30.5070686, 14.2612286, -44.8220367, 44.8065262
21: -42.6714783, 14.9444227, -42.6332016, 14.8943634, -57.5658417, 57.5776253
22: -43.1470490, 17.6184464, -43.0553284, 17.4591637, -60.6062126, 60.6737747
23: -34.3917389, 15.1479836, -34.3407974, 15.1059694, -49.4977074, 49.4887810
24: -36.3923988, 14.8699074, -36.3306046, 14.8382645, -51.2306633, 51.2005119
25: -35.5403023, 17.3383617, -35.4954071, 17.2820072, -52.8223114, 52.8337708
26: -53.2942467, 20.2425404, -53.1780624, 20.0009594, -73.2952042, 73.4206009
27: -36.2102051, 18.9294434, -36.1207199, 18.8830719, -55.0932770, 55.0501633
28: -33.3087921, 18.9929085, -33.2396774, 18.9387016, -52.2474937, 52.2325859
29: -44.8665161, 16.8796310, -44.7976303, 16.7238121, -61.5903282, 61.6772614
30: -42.8404770, 19.9231529, -42.7107773, 19.8495636, -62.6900406, 62.6339302
31: -42.2968788, 15.3397217, -42.2822266, 15.2952871, -57.5921669, 57.6219482
32: -38.4492188, 23.1745968, -38.3954697, 23.1123657, -61.5615845, 61.5700684
33: -48.8468361, 35.9220047, -48.7312660, 35.8753929, -84.7222290, 84.6532745
34: -47.1565857, 21.0624695, -47.0587769, 20.9841576, -68.1041565, 68.0828705
35: -41.6907387, 26.3739471, -41.5811157, 26.3045273, -67.5418091, 67.4882355
36: -42.4083405, 26.6228504, -42.3664474, 26.6014576, -68.1648788, 68.1369629
37: -66.8279572, 22.3003445, -66.7464066, 22.2524872, -86.5638657, 86.5210648
38: -52.5052948, 31.2350559, -52.4573402, 31.1827755, -81.8953018, 81.8822937
39: -60.2577782, 35.4217186, -60.1564026, 35.3614845, -95.6192627, 95.5781250
40: -53.5402451, 28.3264103, -53.4231186, 28.2980385, -81.8382874, 81.7495270
41: -39.0878143, 27.1299057, -39.0430222, 27.0709038, -66.1587219, 66.1729279
42: -32.5319099, 21.9629211, -32.5008659, 21.8975258, -54.4294357, 54.4637871

Time for backsubstitution: 1.72 seconds

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
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4431983
time: 50.47 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4826341
time: 63.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -43.0911217, 35.2124023, -43.0977440, 35.2164383, -78.3075562, 78.3101501
1: -23.3405342, 32.0022583, -23.3432674, 32.0118561, -55.3523903, 55.3455276
2: -18.8317871, 31.8868332, -18.8346825, 31.9112377, -50.7216492, 50.7021294
3: -19.0190983, 35.1062698, -19.0223465, 35.1267090, -53.6989288, 53.6961136
4: -23.4962921, 36.0120583, -23.4939213, 36.0191650, -59.5154572, 59.5059814
5: -21.1865082, 35.4452972, -21.1892300, 35.4696198, -56.0250320, 56.0099869
6: -42.1751404, 26.0705757, -42.1810036, 26.0758457, -68.2509842, 68.2515793
7: -30.4010086, 34.1500778, -30.4058914, 34.1866302, -63.9382935, 63.9153442
8: -29.0115662, 40.0782623, -29.0126686, 40.1040573, -69.1156235, 69.0909271
9: -24.3999062, 31.6535397, -24.4043274, 31.6589661, -54.9370003, 54.9182816
10: -45.8484497, 31.2889061, -45.8642311, 31.2981205, -77.1465683, 77.1531372
11: -48.8858185, 18.1417027, -48.8986130, 18.1431084, -67.0289307, 67.0403137
12: -52.6958466, 18.2045631, -52.7772331, 18.2116604, -69.2443466, 69.2918854
13: -35.7379608, 38.6863251, -35.7592926, 38.6940422, -74.4320068, 74.4456177
14: -78.3702393, 11.0769405, -78.4241943, 11.0807838, -89.4510193, 89.5011368
15: -30.2888565, 30.1230183, -30.3208885, 30.1296253, -60.4184799, 60.4439087
16: -46.2632294, 30.8562088, -46.2744141, 30.8692551, -77.1257248, 77.1237869
17: -77.8306885, 14.7013378, -77.8843002, 14.7061729, -92.5368652, 92.5856400
18: -45.7682838, 21.3216190, -45.7883415, 21.3256760, -67.0939636, 67.1099625
19: -34.5017891, 10.9920959, -34.5105057, 10.9929686, -45.4947586, 45.5026016
20: -30.5850258, 14.3068542, -30.5984421, 14.3091602, -44.8941879, 44.9052963
21: -42.6927414, 14.9492722, -42.7023926, 14.9492598, -57.6420021, 57.6516647
22: -43.2215729, 17.6282368, -43.2645073, 17.6328201, -60.8543930, 60.8927460
23: -34.4107666, 15.1665058, -34.4214096, 15.1678247, -49.5785904, 49.5879135
24: -36.4065590, 14.8863773, -36.4137840, 14.8947144, -51.3012733, 51.3001633
25: -35.5597916, 17.3499146, -35.5681915, 17.3533936, -52.9131851, 52.9181061
26: -53.4323883, 20.2606258, -53.5091476, 20.2665424, -73.6989288, 73.7697754
27: -36.2236404, 18.9482346, -36.2328491, 18.9473286, -55.1709671, 55.1810837
28: -33.3231506, 19.0221996, -33.3319931, 19.0234184, -52.3465691, 52.3541946
29: -44.9258232, 16.8879585, -44.9654770, 16.8897133, -61.8155365, 61.8534355
30: -42.8565674, 19.9881039, -42.8648376, 20.0277672, -62.8843346, 62.8529434
31: -42.3152161, 15.3427734, -42.3235168, 15.3392124, -57.6544266, 57.6662903
32: -38.4834595, 23.1895390, -38.5103149, 23.1948891, -61.6783485, 61.6998520
33: -48.8679161, 35.9455223, -48.8744316, 35.9626160, -84.8305359, 84.8199539
34: -47.1700287, 21.1138935, -47.1776733, 21.1212101, -68.2543869, 68.2562637
35: -41.7062035, 26.4238930, -41.7130432, 26.4282837, -67.6780701, 67.6879959
36: -42.4216461, 26.6371975, -42.4475555, 26.6400719, -68.2135620, 68.2444153
37: -66.8548431, 22.3145485, -66.8647995, 22.3203754, -86.6641083, 86.6640244
38: -52.5158768, 31.2629547, -52.5506592, 31.2690372, -81.9882355, 82.0207672
39: -60.2920914, 35.4406891, -60.2918510, 35.4503937, -95.7424850, 95.7325439
40: -53.5588074, 28.3624763, -53.5648956, 28.3907089, -81.9495163, 81.9273682
41: -39.0987091, 27.1583176, -39.1061630, 27.1603317, -66.2590408, 66.2644806
42: -32.5421371, 21.9877663, -32.5497131, 21.9943256, -54.5364609, 54.5374794

Time for backsubstitution: 1.69 seconds

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
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4431983
time: 60.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4826341
time: 52.57 seconds

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

Time for backsubstitution: 1.79 seconds

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
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
time: 69.99 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.5234558
time: 55.57 seconds

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

Time for backsubstitution: 1.72 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
time: 44.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.5234559, upper bound: 37.5234558
time: 55.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 101.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4431983
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4826341
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4431983
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4826341
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.5234558
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.4795223, upper bound: 37.4802760
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 101.83
Output dim: 8, lower bound: -37.5234559, upper bound: 37.5234558

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -42.9598732, 35.0823364, -42.9594536, 35.0967484, -78.0566254, 78.0417938
1: -23.2530746, 31.9036579, -23.2668953, 31.9136887, -55.1667633, 55.1705551
2: -18.7349796, 31.7439079, -18.7152901, 31.7576866, -50.4723358, 50.4383392
3: -18.9024315, 34.9043884, -18.8672714, 34.9081535, -53.3716049, 53.3249397
4: -23.4244137, 35.9666443, -23.4047260, 35.9494781, -59.3738937, 59.3713684
5: -21.0613556, 35.2288055, -21.0168457, 35.2321243, -55.6661148, 55.6124725
6: -42.1091652, 25.8870068, -42.1025696, 25.8913822, -68.0005493, 67.9895782
7: -30.2382774, 33.9242935, -30.2216358, 33.9514771, -63.5413055, 63.4918747
8: -28.9155502, 39.9334106, -28.9042473, 39.9494553, -68.8650055, 68.8376617
9: -24.2474785, 31.5852242, -24.2115307, 31.5162220, -54.5945129, 54.6464272
10: -45.5675087, 31.1412468, -45.5279617, 31.0433159, -76.6108246, 76.6692047
11: -48.8162308, 18.0481262, -48.8087997, 18.0433235, -66.8595581, 66.8569260
12: -52.2652359, 17.9903851, -52.2792091, 17.8762436, -68.4638062, 68.5931549
13: -35.5735016, 38.6012077, -35.5675240, 38.5551147, -74.1286163, 74.1687317
14: -77.9970169, 10.9530029, -77.9868240, 10.8648415, -88.8618622, 88.9398270
15: -30.0132046, 30.0080109, -30.0462513, 29.9550209, -59.9682236, 60.0542603
16: -46.1451111, 30.7007675, -46.1982193, 30.7372799, -76.8746643, 76.8921051
17: -77.5542450, 14.5595741, -77.5711594, 14.4605389, -92.0147858, 92.1307373
18: -45.6540031, 21.2620163, -45.6483383, 21.2145309, -66.8685303, 66.9103546
19: -34.4335670, 10.9586401, -34.4285431, 10.9523506, -45.3859177, 45.3871841
20: -30.4945126, 14.2647038, -30.4861221, 14.2539396, -44.7484512, 44.7508240
21: -42.6262054, 14.8983202, -42.6165047, 14.8861885, -57.5123940, 57.5148239
22: -42.9336510, 17.4791794, -42.9611244, 17.4472752, -60.3809280, 60.4403038
23: -34.3463821, 15.1022873, -34.3244743, 15.0912724, -49.4376526, 49.4267616
24: -36.3405685, 14.8301783, -36.3144608, 14.8249245, -51.1654930, 51.1446381
25: -35.4498596, 17.2725792, -35.4577255, 17.2702808, -52.7201385, 52.7303047
26: -53.0179749, 20.0650539, -53.0563011, 19.9875202, -73.0054932, 73.1213531
27: -36.1485519, 18.8682766, -36.1059990, 18.8609428, -55.0094948, 54.9742737
28: -33.2611923, 18.9462242, -33.2269592, 18.9238625, -52.1850548, 52.1731834
29: -44.7100792, 16.7652283, -44.7299042, 16.7160110, -61.4260902, 61.4951324
30: -42.7806778, 19.8249550, -42.6931686, 19.8136444, -62.5943222, 62.5181236
31: -42.2711105, 15.2923708, -42.2702217, 15.2848520, -57.5559616, 57.5625916
32: -38.3594131, 23.1144867, -38.3605156, 23.0977936, -61.4572067, 61.4750023
33: -48.7779884, 35.8710480, -48.7162628, 35.8616753, -84.6396637, 84.5873108
34: -47.0903778, 20.9946175, -47.0467758, 20.9602451, -68.0135727, 68.0028381
35: -41.6279602, 26.3169079, -41.5676651, 26.2825890, -67.4556274, 67.4174576
36: -42.3492622, 26.5912399, -42.3490372, 26.5926952, -68.0959320, 68.0889969
37: -66.7495575, 22.2576160, -66.7222519, 22.2406025, -86.4644623, 86.4489059
38: -52.4433899, 31.1830292, -52.4448967, 31.1653023, -81.8093719, 81.8186188
39: -60.1749802, 35.3825684, -60.1291885, 35.3449249, -95.5199051, 95.5117569
40: -53.4568176, 28.2205429, -53.4083443, 28.2511292, -81.7079468, 81.6288910
41: -39.0365372, 27.0323486, -39.0323639, 27.0321045, -66.0686417, 66.0647125
42: -32.4995956, 21.8651924, -32.4905701, 21.8630657, -54.3626633, 54.3557625

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
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
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 696
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
time: 54.97 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
time: 47.21 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -43.0596085, 35.1853104, -42.9737816, 35.1414719, -78.2010803, 78.1590881
1: -23.3247395, 31.9835663, -23.2742062, 31.9459953, -55.2707367, 55.2577744
2: -18.8139477, 31.8379211, -18.7248135, 31.8002205, -50.5940323, 50.5420380
3: -19.0023937, 35.0402222, -18.8769836, 34.9689598, -53.5352478, 53.4724579
4: -23.4686317, 35.9963341, -23.4134998, 35.9594536, -59.4280853, 59.4098358
5: -21.1686630, 35.3740883, -21.0270844, 35.2968903, -55.8405838, 55.7686806
6: -42.1608200, 26.0044231, -42.1116180, 25.9398556, -68.1006775, 68.1160431
7: -30.3783245, 34.0901871, -30.2359352, 34.0259514, -63.7628860, 63.6766739
8: -28.9972458, 40.0390091, -28.9112892, 39.9951630, -68.9924088, 68.9503021
9: -24.3304939, 31.6417160, -24.2442665, 31.5255451, -54.6932373, 54.7511215
10: -45.7285271, 31.2639027, -45.5960312, 31.0577469, -76.7862701, 76.8599319
11: -48.8553047, 18.1041641, -48.8278732, 18.0496178, -66.9049225, 66.9320374
12: -52.5232735, 18.1812363, -52.3978844, 17.8902626, -68.7368622, 68.9068756
13: -35.6750603, 38.6677322, -35.6007042, 38.5688667, -74.2439270, 74.2684326
14: -78.2257080, 11.0687923, -78.0826111, 10.8700523, -89.0957642, 89.1514053
15: -30.1926155, 30.1068726, -30.1170330, 29.9658165, -60.1584320, 60.2239075
16: -46.2244568, 30.8215485, -46.2148247, 30.7846909, -77.0014038, 77.0295105
17: -77.7192993, 14.6829395, -77.6410294, 14.4705868, -92.1898880, 92.3239670
18: -45.7283516, 21.3030605, -45.6742249, 21.2223930, -66.9507446, 66.9772873
19: -34.4787598, 10.9795504, -34.4437370, 10.9549398, -45.4337006, 45.4232864
20: -30.5568409, 14.2951756, -30.5051880, 14.2590647, -44.8159065, 44.8003616
21: -42.6666107, 14.9277096, -42.6308975, 14.8866444, -57.5532532, 57.5586090
22: -43.1222382, 17.6126842, -43.0411758, 17.4564781, -60.5787163, 60.6538620
23: -34.3884583, 15.1352491, -34.3392601, 15.1001396, -49.4885979, 49.4745102
24: -36.3858414, 14.8602772, -36.3275490, 14.8338661, -51.2197075, 51.1878281
25: -35.5318909, 17.3338509, -35.4913864, 17.2799416, -52.8118324, 52.8252373
26: -53.2849808, 20.2366829, -53.1736298, 19.9982281, -73.2832108, 73.4103088
27: -36.2040710, 18.9155807, -36.1178665, 18.8758831, -55.0799561, 55.0334473
28: -33.3060379, 18.9864559, -33.2383614, 18.9355869, -52.2416229, 52.2248154
29: -44.8540535, 16.8750572, -44.7913017, 16.7217293, -61.5757828, 61.6663589
30: -42.8333702, 19.8968735, -42.7074585, 19.8377666, -62.6711349, 62.6043320
31: -42.2931938, 15.3229494, -42.2804756, 15.2874851, -57.5806808, 57.6034241
32: -38.4415817, 23.1707611, -38.3914948, 23.1105194, -61.5521011, 61.5622559
33: -48.8417587, 35.9178505, -48.7288513, 35.8734589, -84.7152176, 84.6466980
34: -47.1538620, 21.0555916, -47.0574875, 20.9807968, -68.0980988, 68.0746384
35: -41.6867714, 26.3702316, -41.5792313, 26.3027744, -67.5364685, 67.4823227
36: -42.3916817, 26.6210384, -42.3586197, 26.6005325, -68.1455994, 68.1260452
37: -66.8157578, 22.2966785, -66.7406769, 22.2508163, -86.5494690, 86.5109863
38: -52.5008698, 31.2295685, -52.4554214, 31.1800327, -81.8878937, 81.8742065
39: -60.2342491, 35.4183540, -60.1453171, 35.3599358, -95.5941849, 95.5636749
40: -53.5321960, 28.3143768, -53.4193840, 28.2925320, -81.8247299, 81.7337646
41: -39.0831909, 27.1188087, -39.0408859, 27.0653419, -66.1485291, 66.1596985
42: -32.5280151, 21.9487553, -32.4990654, 21.8909683, -54.4189835, 54.4478226

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
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
type: B, layer: 1, pos: 648

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
time: 59.79 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
time: 55.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -42.9832306, 35.0963707, -43.0796165, 35.1654968, -78.1487274, 78.1759872
1: -23.2639084, 31.9146194, -23.3336582, 31.9756813, -55.2395897, 55.2482758
2: -18.7483978, 31.7883663, -18.8232517, 31.8666573, -50.5933990, 50.5918846
3: -18.9150009, 34.9622154, -19.0108986, 35.0623207, -53.5275574, 53.5384979
4: -23.4448624, 35.9782143, -23.4808502, 36.0072823, -59.4521446, 59.4590645
5: -21.0751839, 35.2899628, -21.1773453, 35.3985519, -55.8407593, 55.8414917
6: -42.1185913, 25.9301872, -42.1696777, 26.0166702, -68.1352615, 68.0998688
7: -30.2561569, 33.9689140, -30.3895988, 34.0991783, -63.7015152, 63.7141495
8: -28.9248924, 39.9675064, -29.0033188, 40.0558929, -68.9807892, 68.9708252
9: -24.3071899, 31.5936241, -24.3671532, 31.6480446, -54.8277054, 54.8064651
10: -45.6795883, 31.1587887, -45.7924690, 31.2805805, -76.9601669, 76.9512558
11: -48.8391838, 18.0376358, -48.8760605, 18.1106625, -66.9498444, 66.9136963
12: -52.4280396, 18.0101395, -52.6539307, 18.1962872, -68.9577179, 68.9691772
13: -35.6055412, 38.6149826, -35.7087593, 38.6781540, -74.2836914, 74.3237457
14: -78.1296692, 10.9592991, -78.3226089, 11.0747414, -89.2044067, 89.2819061
15: -30.0752964, 30.0191193, -30.2307568, 30.1166000, -60.1918945, 60.2498779
16: -46.1744728, 30.6997032, -46.2532730, 30.8041916, -76.9730377, 76.9462128
17: -77.6527481, 14.5712337, -77.8053207, 14.6930790, -92.3458252, 92.3765564
18: -45.6896858, 21.2753601, -45.7604561, 21.3154602, -67.0051422, 67.0358124
19: -34.4527092, 10.9628353, -34.4934311, 10.9867115, -45.4394226, 45.4562683
20: -30.5182629, 14.2718887, -30.5772877, 14.3019409, -44.8202057, 44.8491745
21: -42.6471558, 14.9028349, -42.6854897, 14.9411612, -57.5883179, 57.5883255
22: -43.0007668, 17.4887981, -43.1628342, 17.6209526, -60.6217194, 60.6516342
23: -34.3652687, 15.1204290, -34.4050522, 15.1527481, -49.5180168, 49.5254822
24: -36.3546219, 14.8452396, -36.3978043, 14.8802843, -51.2349052, 51.2430420
25: -35.4683151, 17.2838974, -35.5296936, 17.3415623, -52.8098755, 52.8135910
26: -53.1531906, 20.0830631, -53.3856239, 20.2532806, -73.4064713, 73.4686890
27: -36.1618652, 18.8845196, -36.2181091, 18.9230804, -55.0849457, 55.1026306
28: -33.2754517, 18.9748611, -33.3192596, 19.0080338, -52.2834854, 52.2941208
29: -44.7686462, 16.7735233, -44.8972168, 16.8818626, -61.6505089, 61.6707382
30: -42.7964478, 19.8882809, -42.8471451, 19.9908028, -62.7872505, 62.7354279
31: -42.2893677, 15.2950335, -42.3113785, 15.3287811, -57.6181488, 57.6064110
32: -38.3934746, 23.1294518, -38.4752541, 23.1803093, -61.5737839, 61.6047058
33: -48.7988892, 35.8942909, -48.8592110, 35.9485626, -84.7474518, 84.7535019
34: -47.1038132, 21.0454178, -47.1657906, 21.0966892, -68.1631851, 68.1756744
35: -41.6433563, 26.3664932, -41.6996384, 26.4061584, -67.5916367, 67.6170654
36: -42.3610115, 26.6055183, -42.4299278, 26.6312447, -68.1436462, 68.1964111
37: -66.7756042, 22.2717266, -66.8401489, 22.3083820, -86.5636444, 86.5912247
38: -52.4536362, 31.2103920, -52.5382271, 31.2512035, -81.9017563, 81.9565735
39: -60.2086296, 35.4013634, -60.2640877, 35.4337196, -95.6423492, 95.6654510
40: -53.4750900, 28.2551155, -53.5499573, 28.3419361, -81.8170242, 81.8050690
41: -39.0471878, 27.0589886, -39.0953636, 27.1199112, -66.1670990, 66.1543503
42: -32.5096703, 21.8894253, -32.5393486, 21.9592724, -54.4689407, 54.4287720

Time for backsubstitution: 1.86 seconds

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
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
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
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 777
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
time: 52.06 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
time: 90.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -43.0829811, 35.2026672, -43.0940628, 35.2117195, -78.2947006, 78.2967300
1: -23.3356171, 31.9961033, -23.3410625, 32.0089226, -55.3445396, 55.3371658
2: -18.8272820, 31.8825836, -18.8326569, 31.9093380, -50.7152252, 50.6956711
3: -19.0148563, 35.0999222, -19.0204315, 35.1238403, -53.6915588, 53.6865425
4: -23.4896145, 36.0081596, -23.4908886, 36.0173874, -59.5070038, 59.4990463
5: -21.1824322, 35.4384689, -21.1873779, 35.4665642, -56.0177460, 56.0006561
6: -42.1704407, 26.0484676, -42.1788864, 26.0658970, -68.2363358, 68.2273560
7: -30.3962002, 34.1420975, -30.4037437, 34.1834221, -63.9307938, 63.9059677
8: -29.0065918, 40.0734634, -29.0104141, 40.1018524, -69.1084442, 69.0838776
9: -24.3910217, 31.6499710, -24.4002743, 31.6573601, -54.9271927, 54.9116783
10: -45.8410110, 31.2814655, -45.8608551, 31.2948036, -77.1358185, 77.1423187
11: -48.8785553, 18.1022835, -48.8953285, 18.1261597, -67.0047150, 66.9976120
12: -52.6867676, 18.2008381, -52.7731895, 18.2099018, -69.2312622, 69.2835083
13: -35.7157898, 38.6817131, -35.7496033, 38.6919556, -74.4077454, 74.4313202
14: -78.3601151, 11.0751381, -78.4196167, 11.0799923, -89.4401093, 89.4947510
15: -30.2612572, 30.1181450, -30.3083000, 30.1274071, -60.3886642, 60.4264450
16: -46.2542496, 30.8274612, -46.2703094, 30.8564281, -77.1039429, 77.0906219
17: -77.8230515, 14.6948109, -77.8806152, 14.7032013, -92.5262527, 92.5754242
18: -45.7644615, 21.3166351, -45.7867432, 21.3233757, -67.0878372, 67.1033783
19: -34.4982109, 10.9837494, -34.5088768, 10.9892263, -45.4874382, 45.4926262
20: -30.5810738, 14.3025169, -30.5966530, 14.3071613, -44.8882370, 44.8991699
21: -42.6879120, 14.9324160, -42.7001839, 14.9417048, -57.6296158, 57.6325989
22: -43.1988831, 17.6225090, -43.2543030, 17.6302299, -60.8291130, 60.8768120
23: -34.4075470, 15.1538115, -34.4199486, 15.1621389, -49.5696869, 49.5737610
24: -36.4001999, 14.8775845, -36.4109306, 14.8907433, -51.2909431, 51.2885132
25: -35.5519257, 17.3455353, -35.5644836, 17.3514137, -52.9033394, 52.9100189
26: -53.4232330, 20.2548962, -53.5049515, 20.2639313, -73.6871643, 73.7598495
27: -36.2175217, 18.9345818, -36.2301254, 18.9409866, -55.1585083, 55.1647072
28: -33.3204613, 19.0158329, -33.3307648, 19.0205460, -52.3410072, 52.3465958
29: -44.9138489, 16.8835411, -44.9593010, 16.8877296, -61.8015785, 61.8428421
30: -42.8495789, 19.9620628, -42.8616982, 20.0170841, -62.8666611, 62.8237610
31: -42.3115501, 15.3260527, -42.3218575, 15.3315096, -57.6430588, 57.6479111
32: -38.4758987, 23.1859322, -38.5063400, 23.1932220, -61.6691208, 61.6922722
33: -48.8631058, 35.9413910, -48.8722229, 35.9607468, -84.8238525, 84.8136139
34: -47.1673164, 21.1070900, -47.1764679, 21.1181221, -68.2486420, 68.2481461
35: -41.7024384, 26.4201794, -41.7112732, 26.4266281, -67.6730347, 67.6823273
36: -42.4050369, 26.6354542, -42.4398079, 26.6392860, -68.1943054, 68.2336884
37: -66.8429031, 22.3109474, -66.8593750, 22.3187103, -86.6501617, 86.6543808
38: -52.5116119, 31.2575150, -52.5487137, 31.2665520, -81.9811935, 82.0125961
39: -60.2688560, 35.4373322, -60.2812805, 35.4489136, -95.7177734, 95.7186127
40: -53.5509453, 28.3505440, -53.5613518, 28.3853874, -81.9363327, 81.9118958
41: -39.0941696, 27.1478958, -39.1041031, 27.1556358, -66.2498016, 66.2519989
42: -32.5382996, 21.9736576, -32.5479851, 21.9879913, -54.5262909, 54.5216446

Time for backsubstitution: 1.86 seconds

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
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
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
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1642
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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
time: 56.25 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
time: 57.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -43.0786705, 35.1149445, -42.9651947, 35.0975685, -78.1762390, 78.0801392
1: -23.3439598, 31.9329453, -23.2692795, 31.9133644, -55.2573242, 55.2022247
2: -18.8526630, 31.8078003, -18.7179813, 31.7774544, -50.6102104, 50.5045242
3: -19.0342007, 34.9781113, -18.8703728, 34.9244003, -53.5199432, 53.3968887
4: -23.4818687, 36.0001640, -23.4064903, 35.9537735, -59.4356422, 59.4066544
5: -21.1978455, 35.3110962, -21.0197067, 35.2516441, -55.8247833, 55.6957893
6: -42.1621475, 25.9476433, -42.1058846, 25.8953190, -68.0574646, 68.0535278
7: -30.4382324, 33.9997711, -30.2256603, 33.9655991, -63.7538452, 63.5717239
8: -29.0356712, 40.0101395, -28.9055576, 39.9690552, -69.0047302, 68.9156952
9: -24.2917366, 31.6435604, -24.2133980, 31.5203667, -54.6437302, 54.7436028
10: -45.6462746, 31.2720947, -45.5415459, 31.0510559, -76.6973267, 76.8136444
11: -48.9230995, 18.1216450, -48.8172302, 18.0321693, -66.9552689, 66.9388733
12: -52.4617844, 18.3109264, -52.3510132, 17.8827343, -68.6601105, 68.9901581
13: -35.6507912, 38.7510300, -35.5730629, 38.5611877, -74.2119751, 74.3240967
14: -78.1638947, 11.1317940, -78.0328674, 10.8677177, -89.0316162, 89.1646576
15: -30.1326313, 30.1581879, -30.0583744, 29.9588490, -60.0914803, 60.2165604
16: -46.2992821, 30.7669487, -46.2048721, 30.7353115, -77.0255127, 76.9620667
17: -77.7030029, 14.7861633, -77.6220551, 14.4626560, -92.1656570, 92.4082184
18: -45.7376671, 21.3491440, -45.6666794, 21.2163048, -66.9539719, 67.0158234
19: -34.4681625, 10.9810638, -34.4347305, 10.9446011, -45.4127655, 45.4157944
20: -30.5630093, 14.3197365, -30.4964752, 14.2558098, -44.8188171, 44.8162117
21: -42.6820793, 14.9291182, -42.6235542, 14.8739052, -57.5559845, 57.5526733
22: -43.0552750, 17.6831512, -42.9883537, 17.4494209, -60.5046959, 60.6715050
23: -34.4100342, 15.1328869, -34.3328209, 15.0925312, -49.5025635, 49.4657059
24: -36.4404297, 14.8746662, -36.3204117, 14.8340816, -51.2745132, 51.1950760
25: -35.4961433, 17.3461838, -35.4594498, 17.2727394, -52.7688828, 52.8056335
26: -53.2213631, 20.3500862, -53.1088181, 19.9903889, -73.2117538, 73.4589081
27: -36.2445755, 18.8840065, -36.1131897, 18.8538475, -55.0984230, 54.9971962
28: -33.3245926, 18.9801083, -33.2339706, 18.9244919, -52.2490845, 52.2140808
29: -44.8300247, 16.9430618, -44.7621803, 16.7173653, -61.5473900, 61.7052422
30: -42.9301224, 19.9368134, -42.6986389, 19.8426838, -62.7728043, 62.6354523
31: -42.3435669, 15.3217726, -42.2758865, 15.2716694, -57.6152344, 57.5976601
32: -38.4484406, 23.2144718, -38.3825417, 23.1021938, -61.5506363, 61.5970154
33: -48.8771591, 35.9478226, -48.7222748, 35.8746338, -84.7517929, 84.6700974
34: -47.1490250, 21.0399284, -47.0519905, 20.9655190, -68.0795593, 68.0533142
35: -41.6830521, 26.3520565, -41.5732651, 26.2866325, -67.5283356, 67.4548492
36: -42.4212799, 26.7457047, -42.3585091, 26.5954342, -68.1717834, 68.2478638
37: -66.8176651, 22.3224258, -66.7274704, 22.2450085, -86.5524216, 86.5201187
38: -52.5726662, 31.3451614, -52.4751091, 31.1700058, -81.9361954, 82.0022736
39: -60.2172928, 35.4384499, -60.1295280, 35.3525848, -95.5698776, 95.5679779
40: -53.5689392, 28.2896423, -53.4124680, 28.2666817, -81.8356171, 81.7021103
41: -39.0908051, 27.0813580, -39.0373230, 27.0345764, -66.1253815, 66.1186829
42: -32.5371933, 21.9348831, -32.4947891, 21.8671169, -54.4043121, 54.4296722

Time for backsubstitution: 1.85 seconds

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
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
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
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 696
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

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
time: 55.58 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
time: 54.38 seconds

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

Time for backsubstitution: 1.77 seconds

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
time: 51.24 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 61.37 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -43.1022835, 35.1294670, -43.0861778, 35.1667633, -78.2690430, 78.2156448
1: -23.3550453, 31.9456959, -23.3362732, 31.9768753, -55.3319206, 55.2819672
2: -18.8664284, 31.8523502, -18.8263340, 31.8866043, -50.7317352, 50.6585884
3: -19.0469170, 35.0389977, -19.0143089, 35.0827751, -53.6802368, 53.6140099
4: -23.5017204, 36.0119896, -23.4833183, 36.0118484, -59.5135689, 59.4953079
5: -21.2120361, 35.3726654, -21.1806717, 35.4219437, -56.0038910, 55.9254494
6: -42.1722641, 25.9923286, -42.1735535, 26.0221748, -68.1944427, 68.1658783
7: -30.4566517, 34.0554161, -30.3940392, 34.1232529, -63.9261017, 63.8054428
8: -29.0452633, 40.0443954, -29.0049171, 40.0758896, -69.1211548, 69.0493164
9: -24.3516121, 31.6525917, -24.3693161, 31.6528339, -54.8774567, 54.9046135
10: -45.7585220, 31.2901230, -45.8063278, 31.2889462, -77.0474701, 77.0964508
11: -48.9469185, 18.1193066, -48.8850899, 18.1103840, -67.0573044, 67.0043945
12: -52.6246262, 18.3306408, -52.7259636, 18.2030239, -69.1544495, 69.3664246
13: -35.6935387, 38.7654839, -35.7234650, 38.6847229, -74.3782654, 74.4889526
14: -78.2968750, 11.1384125, -78.3694611, 11.0778618, -89.3747406, 89.5078735
15: -30.1964912, 30.1699791, -30.2480659, 30.1208878, -60.3173790, 60.4180450
16: -46.3291931, 30.7761688, -46.2605286, 30.8117104, -77.1326218, 77.0270386
17: -77.8020706, 14.7983875, -77.8574600, 14.6957169, -92.4977875, 92.6558456
18: -45.7754250, 21.3628750, -45.7806320, 21.3176155, -67.0930405, 67.1435089
19: -34.4881096, 10.9857817, -34.5005074, 10.9793005, -45.4674110, 45.4862900
20: -30.5871868, 14.3269377, -30.5882015, 14.3040657, -44.8912506, 44.9151382
21: -42.7037125, 14.9345636, -42.6932564, 14.9294844, -57.6331978, 57.6278191
22: -43.1323013, 17.6934280, -43.2002563, 17.6235447, -60.7558441, 60.8936844
23: -34.4290543, 15.1514473, -34.4137154, 15.1544466, -49.5834999, 49.5651627
24: -36.4546204, 14.8912716, -36.4039612, 14.8905706, -51.3451920, 51.2952347
25: -35.5152817, 17.3582001, -35.5320435, 17.3444691, -52.8597488, 52.8902435
26: -53.3596802, 20.3683681, -53.4476357, 20.2567863, -73.6164703, 73.8160019
27: -36.2582054, 18.9005756, -36.2258224, 18.9165840, -55.1747894, 55.1263962
28: -33.3390388, 19.0088272, -33.3265991, 19.0090561, -52.3480949, 52.3354263
29: -44.8897171, 16.9516163, -44.9308472, 16.8834572, -61.7731743, 61.8824615
30: -42.9464378, 20.0008278, -42.8530807, 20.0221863, -62.9686241, 62.8539085
31: -42.3621025, 15.3252249, -42.3178711, 15.3160181, -57.6781197, 57.6430969
32: -38.4828415, 23.2297382, -38.4977341, 23.1851826, -61.6680222, 61.7274704
33: -48.8982849, 35.9714470, -48.8654785, 35.9620552, -84.8603363, 84.8369293
34: -47.1626511, 21.0915298, -47.1711655, 21.1030426, -68.2304001, 68.2272034
35: -41.6988564, 26.4019661, -41.7057266, 26.4105530, -67.6650238, 67.6551819
36: -42.4360046, 26.7599754, -42.4398079, 26.6340561, -68.2206497, 68.3532104
37: -66.8443527, 22.3371735, -66.8458786, 22.3135376, -86.6532669, 86.6616287
38: -52.5845718, 31.3726654, -52.5685844, 31.2564240, -82.0309906, 82.1405411
39: -60.2515640, 35.4577179, -60.2649231, 35.4419098, -95.6934738, 95.7226410
40: -53.5878639, 28.3280964, -53.5545158, 28.3636074, -81.9514694, 81.8826141
41: -39.1020851, 27.1089554, -39.1007385, 27.1235676, -66.2256546, 66.2096939
42: -32.5477066, 21.9608097, -32.5439186, 21.9651260, -54.5128326, 54.5047302

Time for backsubstitution: 1.86 seconds

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
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 647
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

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
time: 57.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
time: 52.31 seconds

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

Time for backsubstitution: 1.85 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 648

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 50.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
time: 55.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 107.81 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4386054
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4786614
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.4765708
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.81
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -42.9496803, 35.0504990, -42.8874130, 35.0249214, -77.9746017, 77.9379120
1: -23.2480946, 31.8743591, -23.2143154, 31.8471031, -55.0951996, 55.0886765
2: -18.7295513, 31.7152615, -18.6584396, 31.6939735, -50.4031296, 50.3527832
3: -18.8976307, 34.8762283, -18.8155479, 34.8426247, -53.3007278, 53.2445335
4: -23.4206161, 35.9479141, -23.3704834, 35.9051361, -59.3257523, 59.3183975
5: -21.0573845, 35.1977310, -20.9631691, 35.1604385, -55.5902481, 55.5266266
6: -42.1002884, 25.8484344, -42.0679245, 25.7981873, -67.8984756, 67.9163589
7: -30.2286072, 33.8652039, -30.1269131, 33.8234100, -63.4050369, 63.3408279
8: -28.9110546, 39.8943214, -28.8380775, 39.8628349, -68.7738876, 68.7323990
9: -24.2417870, 31.5712891, -24.1859341, 31.4794693, -54.5533218, 54.6072006
10: -45.5546875, 31.1276054, -45.4847336, 30.9896469, -76.5443344, 76.6123352
11: -48.7966614, 18.0170593, -48.7648087, 17.9509602, -66.7476196, 66.7818680
12: -52.1910400, 17.9829159, -52.1198196, 17.7472420, -68.2612152, 68.4261780
13: -35.5360641, 38.5888977, -35.4679871, 38.5067368, -74.0428009, 74.0568848
14: -77.9393158, 10.9472542, -77.8501129, 10.7947235, -88.7340393, 88.7973633
15: -29.9450226, 29.9953403, -29.8822632, 29.8883610, -59.8333817, 59.8776016
16: -46.1284485, 30.6190224, -46.1077652, 30.5556831, -76.6765747, 76.7203369
17: -77.4927902, 14.5482025, -77.4341202, 14.3568954, -91.8496857, 91.9823227
18: -45.6315613, 21.2532883, -45.5912170, 21.1819839, -66.8135452, 66.8445053
19: -34.4235687, 10.9537191, -34.4081841, 10.9289179, -45.3524857, 45.3619041
20: -30.4792900, 14.2609444, -30.4405365, 14.2299852, -44.7092743, 44.7014809
21: -42.6130829, 14.8896103, -42.5913506, 14.8481522, -57.4612350, 57.4809608
22: -42.8515511, 17.4697323, -42.7810402, 17.3449535, -60.1965027, 60.2507706
23: -34.3341751, 15.0958424, -34.2907372, 15.0683098, -49.4024849, 49.3865814
24: -36.3266983, 14.8230810, -36.2759628, 14.8013248, -51.1280212, 51.0990448
25: -35.4179306, 17.2654629, -35.3826561, 17.2223091, -52.6402397, 52.6481171
26: -52.9348526, 20.0563393, -52.8698654, 19.8656940, -72.8005447, 72.9262085
27: -36.1345367, 18.8546982, -36.0706482, 18.8237915, -54.9583282, 54.9253464
28: -33.2515030, 18.9423447, -33.1943092, 18.9047546, -52.1562576, 52.1366539
29: -44.6464081, 16.7612247, -44.5838547, 16.6297188, -61.2761269, 61.3450775
30: -42.7656517, 19.7974510, -42.6552124, 19.7429447, -62.5085983, 62.4526634
31: -42.2614059, 15.2810869, -42.2517776, 15.2426310, -57.5040359, 57.5328636
32: -38.3313446, 23.1059761, -38.2927132, 23.0603333, -61.3916779, 61.3986893
33: -48.7681046, 35.8568764, -48.6786270, 35.8164101, -84.5845184, 84.5355072
34: -47.0813446, 20.9846344, -47.0139389, 20.9351788, -67.9794312, 67.9597626
35: -41.6137886, 26.3089638, -41.5282822, 26.2663536, -67.4287567, 67.3707047
36: -42.3160820, 26.5855904, -42.2653961, 26.5388889, -68.0117950, 67.9994507
37: -66.7304840, 22.2463951, -66.6666641, 22.2052040, -86.4063492, 86.3788452
38: -52.4116249, 31.1737747, -52.3614731, 31.1104946, -81.7315750, 81.7292023
39: -60.1620331, 35.3683701, -60.0911102, 35.3048553, -95.4668884, 95.4594803
40: -53.4443817, 28.1707764, -53.3446655, 28.1435127, -81.5878906, 81.5154419
41: -39.0269394, 27.0056496, -38.9963036, 26.9653969, -65.9923401, 66.0019531
42: -32.4908295, 21.8443642, -32.4634361, 21.8019047, -54.2927322, 54.3078003

Time for backsubstitution: 1.85 seconds

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
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1607
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 629

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4097186
time: 55.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4305294
time: 52.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -42.9584465, 35.0784454, -42.9562759, 35.0879173, -78.0463638, 78.0347214
1: -23.2523327, 31.8995438, -23.2652283, 31.9045029, -55.1568375, 55.1647720
2: -18.7343597, 31.7425728, -18.7138519, 31.7547035, -50.4682274, 50.4355812
3: -18.9019203, 34.9026184, -18.8661995, 34.9042702, -53.3632278, 53.3221130
4: -23.4239349, 35.9643555, -23.4037800, 35.9443665, -59.3682098, 59.3681335
5: -21.0609722, 35.2273445, -21.0160370, 35.2288017, -55.6604080, 55.6102371
6: -42.1077881, 25.8807182, -42.0994263, 25.8769913, -67.9847794, 67.9801483
7: -30.2371521, 33.9156113, -30.2191925, 33.9343643, -63.5206070, 63.4805603
8: -28.9149303, 39.9312935, -28.9028950, 39.9453506, -68.8602829, 68.8341904
9: -24.2464771, 31.5833969, -24.2093277, 31.5118484, -54.5893631, 54.6421013
10: -45.5655289, 31.1399212, -45.5234756, 31.0404587, -76.6059875, 76.6633987
11: -48.8133736, 18.0336876, -48.8023911, 18.0112343, -66.8246078, 66.8360748
12: -52.2628174, 17.9897137, -52.2738876, 17.8749352, -68.4600296, 68.5810165
13: -35.5638046, 38.5997200, -35.5467682, 38.5518570, -74.1156616, 74.1464844
14: -77.9928284, 10.9523125, -77.9773560, 10.8632984, -88.8561249, 88.9296722
15: -30.0016613, 30.0060005, -30.0197277, 29.9505672, -59.9522285, 60.0257263
16: -46.1423492, 30.6871948, -46.1919746, 30.7073441, -76.8414841, 76.8723145
17: -77.5502243, 14.5576401, -77.5635529, 14.4561195, -92.0063477, 92.1211929
18: -45.6525536, 21.2605057, -45.6451759, 21.2111492, -66.8637009, 66.9056854
19: -34.4323502, 10.9541273, -34.4258194, 10.9417582, -45.3741074, 45.3799477
20: -30.4930038, 14.2640648, -30.4828396, 14.2523479, -44.7453537, 44.7469025
21: -42.6245728, 14.8918676, -42.6128235, 14.8713112, -57.4958839, 57.5046921
22: -42.9239769, 17.4776573, -42.9404793, 17.4438038, -60.3677826, 60.4181366
23: -34.3452988, 15.0995922, -34.3221207, 15.0856094, -49.4309082, 49.4217148
24: -36.3386269, 14.8271933, -36.3101196, 14.8188744, -51.1575012, 51.1373138
25: -35.4449615, 17.2715225, -35.4470596, 17.2679539, -52.7129135, 52.7185822
26: -53.0118027, 20.0635643, -53.0449600, 19.9840488, -72.9958496, 73.1085205
27: -36.1468391, 18.8623943, -36.1021118, 18.8481979, -54.9950371, 54.9645081
28: -33.2603455, 18.9456005, -33.2251358, 18.9223919, -52.1827393, 52.1707382
29: -44.7048264, 16.7646790, -44.7202377, 16.7147045, -61.4195328, 61.4849167
30: -42.7781525, 19.8166656, -42.6875000, 19.7964172, -62.5745697, 62.5041656
31: -42.2699242, 15.2859859, -42.2674866, 15.2699995, -57.5399246, 57.5534744
32: -38.3573303, 23.1133099, -38.3558922, 23.0954914, -61.4528198, 61.4692001
33: -48.7767830, 35.8697281, -48.7135849, 35.8586807, -84.6354675, 84.5833130
34: -47.0868301, 20.9933319, -47.0387268, 20.9574318, -68.0072632, 67.9940109
35: -41.6218567, 26.3159790, -41.5539246, 26.2804317, -67.4474716, 67.4044418
36: -42.3419228, 26.5907440, -42.3329849, 26.5916557, -68.0874786, 68.0700302
37: -66.7464905, 22.2547989, -66.7154236, 22.2338753, -86.4544907, 86.4414749
38: -52.4382401, 31.1817474, -52.4351349, 31.1623726, -81.8005981, 81.8013763
39: -60.1717529, 35.3814850, -60.1217461, 35.3423042, -95.5140533, 95.5032349
40: -53.4548225, 28.2141418, -53.4037666, 28.2364502, -81.6912689, 81.6179047
41: -39.0352821, 27.0280704, -39.0295944, 27.0223598, -66.0576401, 66.0576630
42: -32.4985809, 21.8612595, -32.4883041, 21.8542576, -54.3528366, 54.3495636

Time for backsubstitution: 1.83 seconds

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
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1607
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
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 629

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4097186
time: 58.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4305294
time: 56.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -43.0495186, 35.1530151, -42.9017601, 35.0692711, -78.1187897, 78.0547791
1: -23.3197174, 31.9533882, -23.2216530, 31.8789215, -55.1986389, 55.1750412
2: -18.8084946, 31.8090591, -18.6679668, 31.7362480, -50.5246048, 50.4562645
3: -18.9976044, 35.0116463, -18.8252640, 34.9030380, -53.4639359, 53.3916893
4: -23.4645748, 35.9773521, -23.3797970, 35.9147072, -59.3792801, 59.3571472
5: -21.1646996, 35.3429184, -20.9733849, 35.2251587, -55.7646484, 55.6826859
6: -42.1517754, 25.9653931, -42.0768509, 25.8459835, -67.9977570, 68.0422440
7: -30.3686771, 34.0275078, -30.1411724, 33.8943863, -63.6217194, 63.5213242
8: -28.9927788, 39.9995613, -28.8451214, 39.9080925, -68.9008713, 68.8446808
9: -24.3245468, 31.6280785, -24.2185707, 31.4885826, -54.6515045, 54.7120285
10: -45.7153664, 31.2502518, -45.5524521, 31.0039597, -76.7193298, 76.8027039
11: -48.8354950, 18.0728798, -48.7836533, 17.9566422, -66.7921371, 66.8565369
12: -52.4488106, 18.1737556, -52.2381477, 17.7612152, -68.5339661, 68.7396088
13: -35.6346512, 38.6555367, -35.4988403, 38.5204620, -74.1551132, 74.1543732
14: -78.1676483, 11.0628891, -77.9454651, 10.7998829, -88.9675293, 89.0083542
15: -30.1238899, 30.0939865, -29.9518051, 29.8989105, -60.0228004, 60.0457916
16: -46.2074661, 30.7391853, -46.1240082, 30.6019020, -76.8017578, 76.8567810
17: -77.6575699, 14.6712074, -77.5038071, 14.3666496, -92.0242157, 92.1750183
18: -45.7056274, 21.2939529, -45.6169128, 21.1894493, -66.8950806, 66.9108658
19: -34.4684830, 10.9746523, -34.4231453, 10.9312582, -45.3997421, 45.3977966
20: -30.5413113, 14.2914238, -30.4592037, 14.2350349, -44.7763443, 44.7506256
21: -42.6531525, 14.9190226, -42.6054840, 14.8482523, -57.5014038, 57.5245056
22: -43.0380898, 17.6030846, -42.8573837, 17.3541336, -60.3922234, 60.4604683
23: -34.3760681, 15.1286716, -34.3053513, 15.0769825, -49.4530487, 49.4340210
24: -36.3718300, 14.8526869, -36.2890434, 14.8095551, -51.1813850, 51.1417313
25: -35.4990845, 17.3266640, -35.4155655, 17.2319450, -52.7310295, 52.7422295
26: -53.2011108, 20.2277260, -52.9862061, 19.8761425, -73.0772552, 73.2139282
27: -36.1901207, 18.9018002, -36.0825958, 18.8385277, -55.0286484, 54.9843979
28: -33.2962456, 18.9823494, -33.2055702, 18.9163475, -52.2125931, 52.1879196
29: -44.7879715, 16.8710537, -44.6438408, 16.6354752, -61.4234467, 61.5148926
30: -42.8180389, 19.8691006, -42.6691895, 19.7664261, -62.5844650, 62.5382919
31: -42.2833099, 15.3112087, -42.2618179, 15.2448483, -57.5281601, 57.5730286
32: -38.4129791, 23.1620636, -38.3231239, 23.0728683, -61.4858475, 61.4851875
33: -48.8318024, 35.9035378, -48.6912689, 35.8279648, -84.6597672, 84.5948029
34: -47.1447372, 21.0451660, -47.0245285, 20.9552765, -68.0634003, 68.0309601
35: -41.6724205, 26.3620148, -41.5395241, 26.2863503, -67.5093155, 67.4350357
36: -42.3574333, 26.6153831, -42.2735596, 26.5467606, -68.0612030, 68.0357208
37: -66.7961121, 22.2853661, -66.6845245, 22.2152290, -86.4906235, 86.4405289
38: -52.4694366, 31.2200565, -52.3713989, 31.1250648, -81.8110275, 81.7838821
39: -60.2209167, 35.4042740, -60.1070786, 35.3196411, -95.5405579, 95.5113525
40: -53.5196457, 28.2641411, -53.3556099, 28.1843853, -81.7040329, 81.6197510
41: -39.0735016, 27.0917835, -39.0046692, 26.9981594, -66.0716629, 66.0964508
42: -32.5191536, 21.9273605, -32.4717712, 21.8292027, -54.3483582, 54.3991318

Time for backsubstitution: 1.85 seconds

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
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 648
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4499398
time: 63.49 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4708954
time: 56.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -43.0582466, 35.1815414, -42.9705811, 35.1327133, -78.1909637, 78.1521225
1: -23.3240108, 31.9799309, -23.2725430, 31.9373245, -55.2613373, 55.2524719
2: -18.8133373, 31.8366280, -18.7233887, 31.7972870, -50.5900116, 50.5393333
3: -19.0019360, 35.0385513, -18.8759766, 34.9650497, -53.5268784, 53.4698105
4: -23.4682293, 35.9941406, -23.4125519, 35.9544106, -59.4226379, 59.4066925
5: -21.1683197, 35.3726463, -21.0262947, 35.2935753, -55.8349075, 55.7664719
6: -42.1594696, 25.9983292, -42.1084824, 25.9255047, -68.0849762, 68.1068115
7: -30.3772774, 34.0831757, -30.2334919, 34.0114632, -63.7449417, 63.6672974
8: -28.9966602, 40.0370216, -28.9099426, 39.9911270, -68.9877853, 68.9469604
9: -24.3295593, 31.6398945, -24.2420635, 31.5211601, -54.6882782, 54.7468987
10: -45.7266083, 31.2626534, -45.5915565, 31.0550041, -76.7816162, 76.8542099
11: -48.8525162, 18.0896797, -48.8215485, 18.0175323, -66.8700485, 66.9112244
12: -52.5209427, 18.1806316, -52.3926086, 17.8890972, -68.7333069, 68.8948364
13: -35.6666489, 38.6663284, -35.5807571, 38.5656433, -74.2322922, 74.2470856
14: -78.2215958, 11.0681686, -78.0732651, 10.8685913, -89.0901871, 89.1414337
15: -30.1814022, 30.1049194, -30.0904770, 29.9613934, -60.1427956, 60.1953964
16: -46.2217751, 30.8083076, -46.2086716, 30.7547722, -76.9683380, 77.0101776
17: -77.7154236, 14.6810532, -77.6335297, 14.4661808, -92.1816025, 92.3145828
18: -45.7269859, 21.3016090, -45.6711655, 21.2190018, -66.9459839, 66.9727783
19: -34.4775925, 10.9749794, -34.4410172, 10.9442816, -45.4218750, 45.4159966
20: -30.5554161, 14.2946148, -30.5019817, 14.2573776, -44.8127937, 44.7965965
21: -42.6650047, 14.9212580, -42.6273117, 14.8717136, -57.5367203, 57.5485687
22: -43.1139603, 17.6112137, -43.0227470, 17.4530296, -60.5669899, 60.6339607
23: -34.3874130, 15.1325874, -34.3369713, 15.0944443, -49.4818573, 49.4695587
24: -36.3839874, 14.8575287, -36.3233032, 14.8278151, -51.2118034, 51.1808319
25: -35.5273285, 17.3328648, -35.4807587, 17.2776070, -52.8049355, 52.8136215
26: -53.2791328, 20.2352161, -53.1624107, 19.9947338, -73.2738647, 73.3976288
27: -36.2023849, 18.9096527, -36.1140747, 18.8631897, -55.0655746, 55.0237274
28: -33.3052216, 18.9858704, -33.2366104, 18.9340744, -52.2392960, 52.2224808
29: -44.8494377, 16.8744984, -44.7816467, 16.7204437, -61.5698814, 61.6561432
30: -42.8309174, 19.8885078, -42.7019119, 19.8224030, -62.6533203, 62.5904198
31: -42.2920303, 15.3172464, -42.2777634, 15.2738504, -57.5658798, 57.5950089
32: -38.4396858, 23.1697292, -38.3872414, 23.1083965, -61.5480804, 61.5569687
33: -48.8405495, 35.9165955, -48.7262459, 35.8705254, -84.7110748, 84.6428375
34: -47.1503906, 21.0543633, -47.0495529, 20.9780102, -68.0918503, 68.0660095
35: -41.6807632, 26.3692741, -41.5655441, 26.3006172, -67.5284119, 67.4693222
36: -42.3848724, 26.6205559, -42.3425446, 26.5995331, -68.1372986, 68.1069031
37: -66.8128204, 22.2938423, -66.7338715, 22.2440681, -86.5396881, 86.5035400
38: -52.4958801, 31.2283039, -52.4456787, 31.1771717, -81.8792953, 81.8569489
39: -60.2310486, 35.4172211, -60.1378479, 35.3572845, -95.5883331, 95.5550690
40: -53.5301971, 28.3081188, -53.4147224, 28.2778530, -81.8080521, 81.7228394
41: -39.0820045, 27.1146889, -39.0380859, 27.0556717, -66.1376801, 66.1527710
42: -32.5270271, 21.9450035, -32.4967880, 21.8822021, -54.4092293, 54.4417915

Time for backsubstitution: 1.77 seconds

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
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
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
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4499398
time: 62.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4708954
time: 52.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -42.9730797, 35.0645447, -43.0076065, 35.0936584, -78.0667419, 78.0721512
1: -23.2589035, 31.8853130, -23.2810402, 31.9089527, -55.1678543, 55.1663513
2: -18.7429466, 31.7596970, -18.7664738, 31.8030128, -50.5242729, 50.5063972
3: -18.9102058, 34.9340134, -18.9593067, 34.9969292, -53.4568176, 53.4582558
4: -23.4408703, 35.9594727, -23.4466972, 35.9629326, -59.4038010, 59.4061699
5: -21.0712605, 35.2588882, -21.1237335, 35.3270721, -55.7651978, 55.7556915
6: -42.1097031, 25.8915749, -42.1351051, 25.9234428, -68.0331421, 68.0266800
7: -30.2465210, 33.9097748, -30.2948799, 33.9718628, -63.5662460, 63.5630493
8: -28.9204254, 39.9284058, -28.9372025, 39.9694519, -68.8898773, 68.8656082
9: -24.3014927, 31.5797234, -24.3416176, 31.6114407, -54.7865906, 54.7673111
10: -45.6667099, 31.1452293, -45.7494659, 31.2271004, -76.8938141, 76.8946991
11: -48.8196106, 18.0066948, -48.8322372, 18.0183754, -66.8379822, 66.8389282
12: -52.3537941, 18.0026855, -52.4947624, 18.0674896, -68.7552948, 68.8024673
13: -35.5679398, 38.6027565, -35.6087952, 38.6287689, -74.1967087, 74.2115479
14: -78.0719299, 10.9535141, -78.1858368, 11.0046310, -89.0765610, 89.1393509
15: -30.0069065, 30.0065269, -30.0678005, 30.0477524, -60.0546570, 60.0743256
16: -46.1578407, 30.6181736, -46.1628189, 30.6220322, -76.7740173, 76.7746811
17: -77.5912247, 14.5598488, -77.6685333, 14.5894489, -92.1806717, 92.2283783
18: -45.6672249, 21.2666073, -45.7032051, 21.2829704, -66.9501953, 66.9698105
19: -34.4427147, 10.9579134, -34.4730568, 10.9636211, -45.4063339, 45.4309692
20: -30.5029926, 14.2681541, -30.5316925, 14.2781343, -44.7811279, 44.7998466
21: -42.6340332, 14.8941736, -42.6602707, 14.9036636, -57.5376968, 57.5544434
22: -42.9186020, 17.4793739, -42.9831543, 17.5186081, -60.4372101, 60.4625282
23: -34.3530960, 15.1139297, -34.3713226, 15.1302099, -49.4833069, 49.4852524
24: -36.3407211, 14.8381424, -36.3589058, 14.8565388, -51.1972580, 51.1970482
25: -35.4363861, 17.2768021, -35.4546318, 17.2935963, -52.7299805, 52.7314339
26: -53.0700150, 20.0743313, -53.1994553, 20.1316204, -73.2016373, 73.2737885
27: -36.1478691, 18.8709240, -36.1826286, 18.8864059, -55.0342751, 55.0535507
28: -33.2657776, 18.9708900, -33.2865944, 18.9889355, -52.2547150, 52.2574844
29: -44.7045021, 16.7694855, -44.7512932, 16.7955475, -61.5000496, 61.5207787
30: -42.7814140, 19.8606720, -42.8092041, 19.9206886, -62.7021027, 62.6698761
31: -42.2796936, 15.2838526, -42.2925148, 15.2867785, -57.5664711, 57.5763664
32: -38.3653679, 23.1209946, -38.4077148, 23.1428871, -61.5082550, 61.5287094
33: -48.7890854, 35.8800735, -48.8216934, 35.9035950, -84.6926804, 84.7017670
34: -47.0947723, 21.0354385, -47.1329193, 21.0716324, -68.1291199, 68.1325912
35: -41.6292114, 26.3585014, -41.6603699, 26.3899689, -67.5647583, 67.5703354
36: -42.3278885, 26.5998878, -42.3454971, 26.5768147, -68.0588684, 68.1066589
37: -66.7565155, 22.2604961, -66.7844391, 22.2728920, -86.5055542, 86.5210114
38: -52.4218979, 31.2011280, -52.4553185, 31.1964111, -81.8239746, 81.8679199
39: -60.1957169, 35.3871193, -60.2259674, 35.3936844, -95.5894012, 95.6130829
40: -53.4626694, 28.2049980, -53.4863319, 28.2345867, -81.6972580, 81.6913300
41: -39.0376434, 27.0322189, -39.0593033, 27.0532036, -66.0908508, 66.0915222
42: -32.5009003, 21.8684025, -32.5122604, 21.8980694, -54.3989716, 54.3806610

Time for backsubstitution: 1.89 seconds

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
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 648
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
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1607
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
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4097186
time: 52.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4706412, upper bound: 37.4305294
time: 58.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -42.9818420, 35.0924568, -43.0764122, 35.1567078, -78.1385498, 78.1688690
1: -23.2631702, 31.9105511, -23.3320141, 31.9665470, -55.2297173, 55.2425652
2: -18.7477779, 31.7870102, -18.8218403, 31.8636894, -50.5892639, 50.5891304
3: -18.9144974, 34.9604263, -19.0098419, 35.0584488, -53.5192108, 53.5357132
4: -23.4443913, 35.9759140, -23.4798431, 36.0021858, -59.4465790, 59.4557571
5: -21.0748291, 35.2885056, -21.1765366, 35.3952599, -55.8351288, 55.8392334
6: -42.1172409, 25.9238949, -42.1665726, 26.0023823, -68.1196213, 68.0904694
7: -30.2550869, 33.9601822, -30.3871250, 34.0826759, -63.6813507, 63.7028503
8: -28.9242935, 39.9654045, -29.0019321, 40.0518036, -68.9760971, 68.9673386
9: -24.3061981, 31.5917816, -24.3649750, 31.6437378, -54.8227158, 54.8021545
10: -45.6776047, 31.1574974, -45.7880096, 31.2777443, -76.9553528, 76.9455109
11: -48.8363190, 18.0232334, -48.8696480, 18.0797825, -66.9160995, 66.8928833
12: -52.4256744, 18.0094833, -52.6485901, 18.1949406, -68.9539337, 68.9570007
13: -35.5958252, 38.6135559, -35.6890030, 38.6749306, -74.2707520, 74.3025589
14: -78.1254883, 10.9586220, -78.3131256, 11.0732365, -89.1987228, 89.2717438
15: -30.0636959, 30.0171700, -30.2044487, 30.1121330, -60.1758270, 60.2216187
16: -46.1717453, 30.6861191, -46.2470322, 30.7741508, -76.9398041, 76.9264145
17: -77.6487579, 14.5692749, -77.7976379, 14.6886692, -92.3374252, 92.3669128
18: -45.6882858, 21.2738380, -45.7572823, 21.3120613, -67.0003510, 67.0311203
19: -34.4515076, 10.9583120, -34.4907188, 10.9761734, -45.4276810, 45.4490318
20: -30.5167675, 14.2712212, -30.5739975, 14.3004856, -44.8172531, 44.8452187
21: -42.6455345, 14.8964148, -42.6818237, 14.9265242, -57.5720596, 57.5782394
22: -42.9910736, 17.4872780, -43.1421051, 17.6175117, -60.6085854, 60.6293831
23: -34.3641930, 15.1177177, -34.4027176, 15.1471653, -49.5113602, 49.5204353
24: -36.3526726, 14.8422337, -36.3935127, 14.8740444, -51.2267151, 51.2357483
25: -35.4634094, 17.2828979, -35.5190964, 17.3392639, -52.8026733, 52.8019943
26: -53.1470299, 20.0815659, -53.3742065, 20.2498646, -73.3968964, 73.4557724
27: -36.1601677, 18.8786697, -36.2142868, 18.9102230, -55.0703888, 55.0929565
28: -33.2746277, 18.9742317, -33.3174591, 19.0066776, -52.2813034, 52.2916908
29: -44.7634125, 16.7729244, -44.8875656, 16.8805428, -61.6439552, 61.6604919
30: -42.7939186, 19.8801308, -42.8414993, 19.9727898, -62.7667084, 62.7216301
31: -42.2881775, 15.2886486, -42.3086777, 15.3141336, -57.6023102, 57.5973282
32: -38.3913879, 23.1282730, -38.4706230, 23.1780014, -61.5693893, 61.5988960
33: -48.7976646, 35.8930283, -48.8565407, 35.9456329, -84.7433014, 84.7495728
34: -47.1002731, 21.0441589, -47.1577377, 21.0938511, -68.1568298, 68.1668472
35: -41.6372337, 26.3655472, -41.6858482, 26.4040031, -67.5834274, 67.6040192
36: -42.3536453, 26.6050739, -42.4140167, 26.6302032, -68.1351700, 68.1776505
37: -66.7725067, 22.2687702, -66.8333664, 22.3017540, -86.5536423, 86.5839539
38: -52.4485016, 31.2090988, -52.5283508, 31.2483101, -81.8930817, 81.9391785
39: -60.2053795, 35.4001961, -60.2566223, 35.4310837, -95.6364594, 95.6568146
40: -53.4730873, 28.2486629, -53.5453491, 28.3273182, -81.8004074, 81.7940140
41: -39.0460205, 27.0546818, -39.0926056, 27.1102219, -66.1562424, 66.1472855
42: -32.5086670, 21.8855171, -32.5370636, 21.9504662, -54.4591331, 54.4225807

Time for backsubstitution: 1.95 seconds

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
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
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
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1607
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4097186
time: 59.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4305294
time: 55.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -43.0728836, 35.1703949, -43.0220985, 35.1395035, -78.2123871, 78.1924896
1: -23.3306465, 31.9659576, -23.2884712, 31.9419098, -55.2725563, 55.2544289
2: -18.8218346, 31.8537254, -18.7759323, 31.8454838, -50.6458740, 50.6100349
3: -19.0100975, 35.0713730, -18.9688320, 35.0580902, -53.6205444, 53.6059341
4: -23.4854927, 35.9891891, -23.4572277, 35.9726830, -59.4581757, 59.4464188
5: -21.1784687, 35.4073257, -21.1337566, 35.3950386, -55.9421234, 55.9147911
6: -42.1614418, 26.0094070, -42.1442261, 25.9720249, -68.1334686, 68.1536331
7: -30.3865910, 34.0794296, -30.3090439, 34.0507050, -63.7890015, 63.7506485
8: -29.0021305, 40.0340080, -28.9443302, 40.0150299, -69.0171585, 68.9783401
9: -24.3850632, 31.6363983, -24.3747025, 31.6205826, -54.8855667, 54.8727112
10: -45.8278503, 31.2679329, -45.8175201, 31.2412586, -77.0691071, 77.0854492
11: -48.8587952, 18.0697937, -48.8513489, 18.0321655, -66.8909607, 66.9211426
12: -52.6123428, 18.1933880, -52.6137161, 18.0811195, -69.0286407, 69.1165619
13: -35.6751480, 38.6695518, -35.6483307, 38.6426010, -74.3177490, 74.3178864
14: -78.3020020, 11.0692406, -78.2824173, 11.0098553, -89.3118591, 89.3516541
15: -30.1923313, 30.1052990, -30.1439152, 30.0583611, -60.2506943, 60.2492142
16: -46.2373238, 30.7452812, -46.1795425, 30.6736393, -76.9039612, 76.9180908
17: -77.7612686, 14.6831532, -77.7434921, 14.5993347, -92.3606033, 92.4266434
18: -45.7417107, 21.3075924, -45.7293320, 21.2905693, -67.0322800, 67.0369263
19: -34.4879150, 10.9788532, -34.4882812, 10.9659472, -45.4538612, 45.4671326
20: -30.5654964, 14.2987938, -30.5507221, 14.2832727, -44.8487701, 44.8495178
21: -42.6744537, 14.9237852, -42.6747589, 14.9038506, -57.5783043, 57.5985451
22: -43.1146889, 17.6129189, -43.0709763, 17.5278397, -60.6425285, 60.6838951
23: -34.3951912, 15.1472254, -34.3860550, 15.1394453, -49.5346375, 49.5332794
24: -36.3861465, 14.8699970, -36.3720169, 14.8666000, -51.2527466, 51.2420120
25: -35.5191422, 17.3383179, -35.4886093, 17.3034840, -52.8226242, 52.8269272
26: -53.3393250, 20.2459240, -53.3180275, 20.1420059, -73.4813309, 73.5639496
27: -36.2035866, 18.9207878, -36.1946869, 18.9040718, -55.1076584, 55.1154747
28: -33.3106689, 19.0117378, -33.2979469, 19.0014038, -52.3120728, 52.3096848
29: -44.8476486, 16.8795242, -44.8119507, 16.8014774, -61.6491241, 61.6914749
30: -42.8342590, 19.9339371, -42.8234978, 19.9455414, -62.7798004, 62.7574348
31: -42.3017197, 15.3143158, -42.3028374, 15.2891417, -57.5908623, 57.6171532
32: -38.4472961, 23.1772938, -38.4382553, 23.1556091, -61.6029053, 61.6155472
33: -48.8532410, 35.9270287, -48.8347549, 35.9155006, -84.7687378, 84.7617798
34: -47.1582718, 21.0966549, -47.1435127, 21.0926189, -68.2140350, 68.2044983
35: -41.6881256, 26.4119835, -41.6717110, 26.4103088, -67.6459122, 67.6351013
36: -42.3708954, 26.6298199, -42.3547440, 26.5849190, -68.1093979, 68.1432037
37: -66.8232574, 22.2996407, -66.8032227, 22.2830734, -86.5912628, 86.5839386
38: -52.4802246, 31.2479839, -52.4654007, 31.2115841, -81.9042664, 81.9231873
39: -60.2555656, 35.4233093, -60.2430344, 35.4086800, -95.6642456, 95.6663437
40: -53.5384560, 28.3001080, -53.4976845, 28.2773991, -81.8158569, 81.7977905
41: -39.0844803, 27.1208096, -39.0679359, 27.0885391, -66.1730194, 66.1887436
42: -32.5293999, 21.9521446, -32.5207443, 21.9262505, -54.4556503, 54.4728889

Time for backsubstitution: 1.84 seconds

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
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 648
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4499398
time: 67.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4708954
time: 55.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -43.0816345, 35.1989098, -43.0908966, 35.2029648, -78.2845993, 78.2898102
1: -23.3349152, 31.9924545, -23.3393688, 32.0002060, -55.3351212, 55.3318253
2: -18.8266716, 31.8812943, -18.8312454, 31.9063988, -50.7111435, 50.6930084
3: -19.0144005, 35.0982666, -19.0194168, 35.1199379, -53.6832886, 53.6838684
4: -23.4891987, 36.0059662, -23.4899044, 36.0123329, -59.5015335, 59.4958725
5: -21.1820965, 35.4370575, -21.1866016, 35.4632950, -56.0121613, 55.9984589
6: -42.1691132, 26.0423355, -42.1758041, 26.0516129, -68.2207260, 68.2181396
7: -30.3951149, 34.1350975, -30.4012947, 34.1687012, -63.9127350, 63.8965683
8: -29.0060005, 40.0714951, -29.0090809, 40.0978127, -69.1038132, 69.0805740
9: -24.3900757, 31.6481323, -24.3980637, 31.6530495, -54.9223938, 54.9074173
10: -45.8390961, 31.2802544, -45.8563919, 31.2920685, -77.1311646, 77.1366425
11: -48.8757782, 18.0883713, -48.8890076, 18.0952911, -66.9710693, 66.9773788
12: -52.6844673, 18.2002335, -52.7678185, 18.2087307, -69.2277374, 69.2714386
13: -35.7073364, 38.6803055, -35.7297783, 38.6887512, -74.3960876, 74.4100800
14: -78.3560715, 11.0745163, -78.4102325, 11.0785084, -89.4345779, 89.4847488
15: -30.2500381, 30.1162148, -30.2819443, 30.1230183, -60.3730545, 60.3981590
16: -46.2515526, 30.8142071, -46.2641830, 30.8264198, -77.0708008, 77.0712433
17: -77.8191757, 14.6929913, -77.8730545, 14.6988583, -92.5180359, 92.5660477
18: -45.7630730, 21.3151932, -45.7836342, 21.3199730, -67.0830460, 67.0988312
19: -34.4970398, 10.9791937, -34.5061684, 10.9786892, -45.4757309, 45.4853630
20: -30.5796623, 14.3018856, -30.5934258, 14.3055897, -44.8852539, 44.8953094
21: -42.6863174, 14.9259558, -42.6965866, 14.9270277, -57.6133461, 57.6225433
22: -43.1917419, 17.6210442, -43.2375946, 17.6267929, -60.8185349, 60.8586388
23: -34.4065247, 15.1511440, -34.4176598, 15.1565275, -49.5630531, 49.5688019
24: -36.3983688, 14.8748169, -36.4066925, 14.8845367, -51.2829056, 51.2815094
25: -35.5473976, 17.3445339, -35.5539093, 17.3490982, -52.8964958, 52.8984451
26: -53.4174042, 20.2534027, -53.4936790, 20.2604752, -73.6778793, 73.7470856
27: -36.2158508, 18.9286842, -36.2263412, 18.9284973, -55.1443481, 55.1550255
28: -33.3196487, 19.0152588, -33.3290100, 19.0192089, -52.3388596, 52.3442688
29: -44.9092712, 16.8829765, -44.9496689, 16.8864021, -61.7956734, 61.8326454
30: -42.8471222, 19.9542542, -42.8561745, 20.0010834, -62.8482056, 62.8104286
31: -42.3104057, 15.3203306, -42.3191299, 15.3178406, -57.6282463, 57.6394615
32: -38.4740067, 23.1849384, -38.5020752, 23.1910439, -61.6650505, 61.6870117
33: -48.8619232, 35.9401169, -48.8696442, 35.9577789, -84.8197021, 84.8097610
34: -47.1638412, 21.1058693, -47.1685257, 21.1153030, -68.2423401, 68.2395172
35: -41.6963959, 26.4192638, -41.6975708, 26.4244766, -67.6649475, 67.6693344
36: -42.3982277, 26.6350117, -42.4238968, 26.6382465, -68.1860580, 68.2147217
37: -66.8399963, 22.3080902, -66.8526077, 22.3120575, -86.6403580, 86.6470947
38: -52.5066414, 31.2562351, -52.5388985, 31.2637119, -81.9726105, 81.9953156
39: -60.2657089, 35.4362488, -60.2737885, 35.4463043, -95.7120132, 95.7100372
40: -53.5489960, 28.3442993, -53.5567436, 28.3707790, -81.9197769, 81.9010468
41: -39.0929832, 27.1438141, -39.1013527, 27.1460018, -66.2389832, 66.2451630
42: -32.5373116, 21.9698792, -32.5457458, 21.9792385, -54.5165482, 54.5156250

Time for backsubstitution: 1.91 seconds

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
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4499398
time: 59.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4708954
time: 62.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -43.0687485, 35.0828857, -42.8931351, 35.0253487, -78.0941010, 77.9760208
1: -23.3389969, 31.9034157, -23.2166901, 31.8461647, -55.1851616, 55.1201057
2: -18.8472538, 31.7790203, -18.6610794, 31.7134819, -50.5407753, 50.4187279
3: -19.0293770, 34.9491920, -18.8185234, 34.8575745, -53.4480896, 53.3159103
4: -23.4779263, 35.9813766, -23.3721581, 35.9091377, -59.3870621, 59.3535347
5: -21.1938972, 35.2793388, -20.9658966, 35.1786652, -55.7478333, 55.6092644
6: -42.1531677, 25.9086094, -42.0711212, 25.8011875, -67.9543533, 67.9797287
7: -30.4287033, 33.9396667, -30.1308117, 33.8351021, -63.6153870, 63.4200439
8: -29.0311546, 39.9708366, -28.8393211, 39.8820953, -68.9132538, 68.8101578
9: -24.2860432, 31.6297874, -24.1876984, 31.4832287, -54.6013489, 54.7043152
10: -45.6333733, 31.2584152, -45.4980698, 30.9969482, -76.6303253, 76.7564850
11: -48.9033356, 18.0901966, -48.7729607, 17.9390411, -66.8423767, 66.8631592
12: -52.3873253, 18.3033848, -52.1908646, 17.7535210, -68.4570999, 68.8227844
13: -35.6125641, 38.7386627, -35.4726906, 38.5126877, -74.1252518, 74.2113495
14: -78.1048813, 11.1259117, -77.8954620, 10.7975731, -88.9024506, 89.0213776
15: -30.0637512, 30.1454849, -29.8927670, 29.8920002, -59.9557495, 60.0382538
16: -46.2825546, 30.6842957, -46.1141434, 30.5525856, -76.8261185, 76.7890778
17: -77.6385880, 14.7748203, -77.4830704, 14.3587914, -91.9973755, 92.2578888
18: -45.7146530, 21.3403740, -45.6088142, 21.1833992, -66.8980560, 66.9491882
19: -34.4580765, 10.9762907, -34.4142418, 10.9207916, -45.3788681, 45.3905334
20: -30.5475864, 14.3161745, -30.4505367, 14.2317200, -44.7793045, 44.7667122
21: -42.6688232, 14.9196320, -42.5981712, 14.8342381, -57.5030594, 57.5178032
22: -42.9729156, 17.6735802, -42.8075027, 17.3470840, -60.3199997, 60.4810829
23: -34.3977852, 15.1256924, -34.2989235, 15.0686474, -49.4664307, 49.4246140
24: -36.4266586, 14.8660030, -36.2818146, 14.8085146, -51.2351723, 51.1478195
25: -35.4641762, 17.3389664, -35.3840675, 17.2246838, -52.6888580, 52.7230339
26: -53.1334534, 20.3413887, -52.9190865, 19.8683910, -73.0018463, 73.2604752
27: -36.2309189, 18.8700771, -36.0777664, 18.8159409, -55.0468597, 54.9478455
28: -33.3148346, 18.9759426, -33.2011299, 18.9051971, -52.2200317, 52.1770706
29: -44.7628555, 16.9390488, -44.6138306, 16.6310730, -61.3939285, 61.5528793
30: -42.9150009, 19.9073677, -42.6603622, 19.7698898, -62.6848907, 62.5677299
31: -42.3339272, 15.3104391, -42.2573318, 15.2291212, -57.5630493, 57.5677719
32: -38.4204178, 23.2057457, -38.3145370, 23.0642681, -61.4846878, 61.5202827
33: -48.8671188, 35.9336128, -48.6844788, 35.8292160, -84.6963348, 84.6180878
34: -47.1397476, 21.0294971, -47.0188904, 20.9400291, -68.0447388, 68.0094528
35: -41.6686516, 26.3439484, -41.5333786, 26.2704544, -67.5012360, 67.4073334
36: -42.3862534, 26.7402763, -42.2731018, 26.5416908, -68.0867920, 68.1574402
37: -66.7984009, 22.3112068, -66.6714935, 22.2092857, -86.4937744, 86.4507141
38: -52.5385170, 31.3358517, -52.3881989, 31.1150913, -81.8561020, 81.9089661
39: -60.2043076, 35.4244308, -60.0913086, 35.3122292, -95.5165405, 95.5157394
40: -53.5566254, 28.2386074, -53.3486328, 28.1578102, -81.7144318, 81.5872421
41: -39.0811577, 27.0545273, -39.0011520, 26.9672050, -66.0483627, 66.0556793
42: -32.5283394, 21.9136200, -32.4675369, 21.8051987, -54.3335381, 54.3811569

Time for backsubstitution: 1.91 seconds

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
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
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
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1607
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
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4478014
time: 57.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4687158
time: 76.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -43.0772896, 35.1111832, -42.9620361, 35.0888214, -78.1661072, 78.0732193
1: -23.3432198, 31.9290791, -23.2676277, 31.9042969, -55.2475166, 55.1967087
2: -18.8520870, 31.8065071, -18.7165985, 31.7744980, -50.6061478, 50.5017929
3: -19.0337257, 34.9764023, -18.8693523, 34.9205322, -53.5116501, 53.3941498
4: -23.4814720, 35.9979897, -23.4055557, 35.9486771, -59.4294662, 59.4035454
5: -21.1975079, 35.3096199, -21.0189323, 35.2483177, -55.8191071, 55.6935616
6: -42.1607971, 25.9416237, -42.1027794, 25.8809624, -68.0417633, 68.0444031
7: -30.4371872, 33.9926453, -30.2232265, 33.9523468, -63.7367020, 63.5620193
8: -29.0350494, 40.0081940, -28.9041805, 39.9650803, -69.0001297, 68.9123764
9: -24.2907753, 31.6418743, -24.2112217, 31.5160217, -54.6382484, 54.7394600
10: -45.6443100, 31.2708855, -45.5370293, 31.0482788, -76.6925888, 76.8079147
11: -48.9203110, 18.1071892, -48.8108215, 18.0001106, -66.9204254, 66.9180145
12: -52.4594688, 18.3103085, -52.3457451, 17.8814201, -68.6564560, 68.9782104
13: -35.6414642, 38.7496338, -35.5523376, 38.5579567, -74.1994171, 74.3019714
14: -78.1608276, 11.1311436, -78.0254822, 10.8662376, -89.0270691, 89.1566238
15: -30.1211853, 30.1562614, -30.0319004, 29.9544106, -60.0755959, 60.1881638
16: -46.2966042, 30.7538948, -46.1986465, 30.7063694, -76.9934082, 76.9427795
17: -77.7003250, 14.7842903, -77.6158600, 14.4583015, -92.1586304, 92.4001465
18: -45.7363167, 21.3476601, -45.6636200, 21.2128601, -66.9491730, 67.0112762
19: -34.4669991, 10.9766436, -34.4320450, 10.9347477, -45.4017487, 45.4086876
20: -30.5615902, 14.3192692, -30.4932423, 14.2541666, -44.8157578, 44.8125114
21: -42.6805153, 14.9231033, -42.6199379, 14.8589783, -57.5394936, 57.5430412
22: -43.0459137, 17.6816483, -42.9687119, 17.4459801, -60.4918938, 60.6503601
23: -34.4090195, 15.1305952, -34.3304634, 15.0868511, -49.4958725, 49.4610596
24: -36.4385605, 14.8724575, -36.3161011, 14.8287363, -51.2672958, 51.1885605
25: -35.4913254, 17.3451996, -35.4487991, 17.2704964, -52.7618217, 52.7939987
26: -53.2180328, 20.3485794, -53.1005363, 19.9869099, -73.2049408, 73.4491119
27: -36.2429543, 18.8781624, -36.1093559, 18.8411293, -55.0840836, 54.9875183
28: -33.3238106, 18.9795151, -33.2321548, 18.9230385, -52.2468491, 52.2116699
29: -44.8254166, 16.9425011, -44.7525902, 16.7160721, -61.5414886, 61.6950912
30: -42.9276962, 19.9279900, -42.6930084, 19.8273277, -62.7550240, 62.6209984
31: -42.3424072, 15.3156729, -42.2731476, 15.2568274, -57.5992355, 57.5888214
32: -38.4464111, 23.2134418, -38.3779335, 23.0999222, -61.5463333, 61.5913773
33: -48.8760338, 35.9465027, -48.7196083, 35.8717308, -84.7477646, 84.6661072
34: -47.1456413, 21.0386620, -47.0439682, 20.9627171, -68.0733948, 68.0445404
35: -41.6772346, 26.3511295, -41.5595627, 26.2844944, -67.5204163, 67.4418793
36: -42.4145432, 26.7452812, -42.3425140, 26.5944080, -68.1637115, 68.2286377
37: -66.8147125, 22.3195038, -66.7207336, 22.2382355, -86.5427094, 86.5111694
38: -52.5688248, 31.3439331, -52.4664917, 31.1671085, -81.9288559, 81.9858932
39: -60.2140846, 35.4372902, -60.1220741, 35.3499222, -95.5640106, 95.5593643
40: -53.5670319, 28.2834225, -53.4078331, 28.2519970, -81.8190308, 81.6912537
41: -39.0896149, 27.0772972, -39.0345573, 27.0248127, -66.1144257, 66.1118546
42: -32.5362015, 21.9312782, -32.4925003, 21.8583450, -54.3945465, 54.4237785

Time for backsubstitution: 1.87 seconds

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
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1653
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
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4478014
time: 61.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4687158
time: 63.43 seconds

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

Time for backsubstitution: 1.90 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 616

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
time: 61.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
time: 57.21 seconds

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

Time for backsubstitution: 1.85 seconds

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
time: 61.15 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
time: 60.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -43.0923691, 35.0973816, -43.0141296, 35.0946045, -78.1869736, 78.1115112
1: -23.3500977, 31.9161339, -23.2836285, 31.9098282, -55.2599258, 55.1997604
2: -18.8610115, 31.8235779, -18.7695694, 31.8227673, -50.6624680, 50.5729523
3: -19.0421181, 35.0100708, -18.9626350, 35.0162277, -53.6086731, 53.5333061
4: -23.4975548, 35.9932022, -23.4491367, 35.9672508, -59.4648056, 59.4423370
5: -21.2081070, 35.3409195, -21.1270142, 35.3491478, -55.9272079, 55.8391266
6: -42.1633034, 25.9532948, -42.1388931, 25.9281044, -68.0914078, 68.0921860
7: -30.4471226, 33.9943314, -30.2992477, 33.9921570, -63.7871399, 63.6528244
8: -29.0407734, 40.0050659, -28.9387589, 39.9891319, -69.0299072, 68.9438248
9: -24.3459358, 31.6388130, -24.3437061, 31.6159172, -54.8352928, 54.8653984
10: -45.7456207, 31.2765369, -45.7631302, 31.2351494, -76.9807739, 77.0396652
11: -48.9271355, 18.0880623, -48.8410568, 18.0160637, -66.9431992, 66.9291229
12: -52.5501938, 18.3231602, -52.5661240, 18.0740852, -68.9516907, 69.1993637
13: -35.6541252, 38.7531815, -35.6226807, 38.6352310, -74.2893524, 74.3758621
14: -78.2377930, 11.1325645, -78.2320938, 11.0077610, -89.2455521, 89.3646545
15: -30.1275177, 30.1573029, -30.0836525, 30.0518894, -60.1794052, 60.2409554
16: -46.3125000, 30.6937523, -46.1698685, 30.6286621, -76.9331589, 76.8544159
17: -77.7375488, 14.7870388, -77.7188721, 14.5919514, -92.3294983, 92.5059128
18: -45.7523613, 21.3541508, -45.7227821, 21.2849045, -67.0372620, 67.0769348
19: -34.4780273, 10.9810047, -34.4800262, 10.9559927, -45.4340210, 45.4610291
20: -30.5717144, 14.3233986, -30.5423203, 14.2800894, -44.8518028, 44.8657188
21: -42.6904640, 14.9250507, -42.6678658, 14.8902464, -57.5807114, 57.5929184
22: -43.0498199, 17.6838646, -43.0199242, 17.5212021, -60.5710220, 60.7037888
23: -34.4168320, 15.1442060, -34.3798370, 15.1307421, -49.5475731, 49.5240440
24: -36.4408951, 14.8825579, -36.3649902, 14.8652382, -51.3061333, 51.2475471
25: -35.4833145, 17.3509960, -35.4566879, 17.2965221, -52.7798386, 52.8076859
26: -53.2716751, 20.3597240, -53.2583542, 20.1349964, -73.4066696, 73.6180801
27: -36.2445908, 18.8866634, -36.1902504, 18.8790627, -55.1236534, 55.0769119
28: -33.3292770, 19.0045967, -33.2937775, 18.9898472, -52.3191223, 52.2983742
29: -44.8225021, 16.9475956, -44.7827797, 16.7971649, -61.6196671, 61.7303772
30: -42.9312820, 19.9709053, -42.8148651, 19.9489326, -62.8802147, 62.7857704
31: -42.3524857, 15.3139677, -42.2989616, 15.2737236, -57.6262093, 57.6129303
32: -38.4548035, 23.2211189, -38.4300613, 23.1473808, -61.6021843, 61.6511803
33: -48.8882904, 35.9572029, -48.8277931, 35.9168625, -84.8051529, 84.7849960
34: -47.1533775, 21.0811405, -47.1380768, 21.0776825, -68.1957703, 68.1833725
35: -41.6845016, 26.3938560, -41.6659470, 26.3943634, -67.6379318, 67.6078110
36: -42.4009171, 26.7545509, -42.3544159, 26.5796700, -68.1349945, 68.2628021
37: -66.8251114, 22.3259182, -66.7899017, 22.2778072, -86.5946198, 86.5924149
38: -52.5504761, 31.3633957, -52.4822922, 31.2014771, -81.9509430, 82.0479965
39: -60.2385979, 35.4436951, -60.2266922, 35.4015808, -95.6401825, 95.6703873
40: -53.5755310, 28.2768288, -53.4908447, 28.2549553, -81.8304901, 81.7676697
41: -39.0924759, 27.0820770, -39.0645905, 27.0562592, -66.1487350, 66.1466675
42: -32.5388794, 21.9393311, -32.5167122, 21.9032669, -54.4421463, 54.4560432

Time for backsubstitution: 1.78 seconds

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
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
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

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4478014
time: 55.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4687158
time: 67.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -43.1009254, 35.1257019, -43.0830040, 35.1580276, -78.2589569, 78.2087097
1: -23.3543282, 31.9418125, -23.3346119, 31.9678345, -55.3221626, 55.2764244
2: -18.8658218, 31.8510780, -18.8249702, 31.8836479, -50.7276688, 50.6558952
3: -19.0464516, 35.0372925, -19.0132751, 35.0789642, -53.6719894, 53.6112900
4: -23.5013008, 36.0098305, -23.4823818, 36.0067825, -59.5080833, 59.4922104
5: -21.2117043, 35.3712311, -21.1799068, 35.4186325, -55.9982681, 55.9233170
6: -42.1709213, 25.9862747, -42.1704826, 26.0078354, -68.1787567, 68.1567535
7: -30.4555473, 34.0483246, -30.3915958, 34.1097870, -63.9087372, 63.7957306
8: -29.0446453, 40.0424271, -29.0035439, 40.0718842, -69.1165314, 69.0459747
9: -24.3506699, 31.6509018, -24.3671341, 31.6485367, -54.8721542, 54.9004326
10: -45.7565689, 31.2889366, -45.8018608, 31.2861729, -77.0427399, 77.0907974
11: -48.9441185, 18.1047935, -48.8787079, 18.0795383, -67.0236588, 66.9835052
12: -52.6223297, 18.3300629, -52.7206650, 18.2016888, -69.1507950, 69.3543854
13: -35.6845169, 38.7641144, -35.7036781, 38.6815491, -74.3660660, 74.4677887
14: -78.2938385, 11.1377831, -78.3619690, 11.0763874, -89.3702240, 89.4997559
15: -30.1850910, 30.1680679, -30.2217979, 30.1164684, -60.3015594, 60.3898659
16: -46.3265343, 30.7630539, -46.2543030, 30.7826424, -77.1004639, 77.0077515
17: -77.7993927, 14.7965164, -77.8512497, 14.6913471, -92.4907379, 92.6477661
18: -45.7740974, 21.3614292, -45.7775574, 21.3142319, -67.0883331, 67.1389847
19: -34.4869385, 10.9813356, -34.4978371, 10.9694281, -45.4563675, 45.4791718
20: -30.5857754, 14.3264742, -30.5849571, 14.3025131, -44.8882904, 44.9114304
21: -42.7021637, 14.9285088, -42.6896515, 14.9148445, -57.6170082, 57.6181602
22: -43.1229324, 17.6919441, -43.1821976, 17.6201115, -60.7430420, 60.8741417
23: -34.4280243, 15.1491270, -34.4113579, 15.1488438, -49.5768661, 49.5604858
24: -36.4527740, 14.8890705, -36.3996964, 14.8852959, -51.3380699, 51.2887650
25: -35.5104904, 17.3572350, -35.5214462, 17.3422661, -52.8527565, 52.8786812
26: -53.3563614, 20.3668861, -53.4393234, 20.2533379, -73.6096954, 73.8062134
27: -36.2566071, 18.8947029, -36.2220306, 18.9037704, -55.1603775, 55.1167336
28: -33.3382416, 19.0082493, -33.3248062, 19.0077477, -52.3459892, 52.3330536
29: -44.8851433, 16.9510536, -44.9212685, 16.8821869, -61.7673302, 61.8723221
30: -42.9440079, 19.9919510, -42.8474884, 20.0061436, -62.9501495, 62.8394394
31: -42.3609428, 15.3190689, -42.3151627, 15.3013372, -57.6622810, 57.6342316
32: -38.4808044, 23.2287369, -38.4931221, 23.1829033, -61.6637077, 61.7218590
33: -48.8971825, 35.9701691, -48.8628426, 35.9591141, -84.8562927, 84.8330078
34: -47.1592102, 21.0903187, -47.1631317, 21.1002617, -68.2242813, 68.2184372
35: -41.6929817, 26.4010277, -41.6919250, 26.4084015, -67.6570740, 67.6422501
36: -42.4292831, 26.7595310, -42.4239807, 26.6330204, -68.2125549, 68.3341980
37: -66.8414536, 22.3342209, -66.8391647, 22.3068638, -86.6435089, 86.6528397
38: -52.5807457, 31.3714733, -52.5599670, 31.2535706, -82.0236893, 82.1242371
39: -60.2483368, 35.4565048, -60.2574654, 35.4392700, -95.6876068, 95.7139740
40: -53.5859070, 28.3218899, -53.5499229, 28.3489838, -81.9348907, 81.8718109
41: -39.1008949, 27.1048641, -39.0980186, 27.1138668, -66.2147598, 66.2028809
42: -32.5467186, 21.9571400, -32.5416451, 21.9563637, -54.5030823, 54.4987869

Time for backsubstitution: 1.89 seconds

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
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1016
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

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4478014
time: 55.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4687158
time: 57.11 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 115.02 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4097186
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4305294
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4097186
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4305294
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4499398
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4708954
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4499398
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4708954
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4097186
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4706412, upper bound: 37.4305294
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4097186
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4305294
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4499398
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4708954
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4499398
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4708954
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4478014
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4687158
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4478014
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4687158
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4915079
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.5124481
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4478014
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4254963, upper bound: 37.4687158
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4478014
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 115.02
Output dim: 8, lower bound: -37.4690362, upper bound: 37.4687158
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 115.02
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 115.02
Output dim: 8, lower bound: -37.4336815, upper bound: 37.5203043

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 82.60 + 3537.04 = 3619.64 seconds
