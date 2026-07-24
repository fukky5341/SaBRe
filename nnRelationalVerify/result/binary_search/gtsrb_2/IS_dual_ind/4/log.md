## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 148.163763758
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613)
1: (-112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889)
2: (-99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441)
3: (-115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597)
4: (-122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602)
5: (-111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894)
6: (-137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438)
7: (-135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221)
8: (-127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125)
9: (-119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873)
10: (-174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921)
11: (-166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880)
12: (-156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342)
13: (-161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905)
14: (-238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144)
15: (-139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779)
16: (-182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739)
17: (-249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059)
18: (-152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643)
19: (-117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732)
20: (-110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466)
21: (-149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598)
22: (-158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844)
23: (-120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706)
24: (-147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244)
25: (-124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114)
26: (-171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330)
27: (-155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779)
28: (-116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059)
29: (-174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674)
30: (-146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513)
31: (-149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037)
32: (-146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617)
33: (-192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861)
34: (-161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885)
35: (-163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379)
36: (-154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494)
37: (-223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008)
38: (-190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223)
39: (-226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820)
40: (-192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042)
41: (-144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738)
42: (-111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497)

## BASE Result
execution time: IAR + LP analysis = 2.77 + 758.36 = 761.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -151.4425369, upper bound: 151.4425371


# Binary Search by BASE starts (time budget: 17238.87 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=233.43865966796875
rel_dist={3: [-146.7491330209824, 146.7491331173323]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=233.43865966796875
rel_dist={3: [-149.40356280758328, 149.4035629176625]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=233.43865966796875
rel_dist={3: [-147.729598459751, 147.72959856394056]}

## Binary search (step 3) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=8, k_high=8, k_mid=8, eps_mid=0.0312500, abs_max=233.43865966796875
rel_dist={3: [-148.6090126930012, 148.60901277819403]}

## Binary Search Result
Binary search time: 3759.08 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.02734375


# Individual Split (IS_dual_ind) starts
Time budget: 13479.79 seconds

## Binary search (step 0) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1162
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1225
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1175
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1178
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 257
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 321
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 336
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 1792
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 640

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
time: 799.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
time: 881.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1680.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1680.86
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1680.86
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -193.7549591, 123.4849701, -194.2824097, 123.7356949, -317.4906616, 317.7673950
1: -112.2062836, 98.0295868, -112.5968475, 98.2420578, -210.4483337, 210.6264343
2: -98.4843063, 90.5573730, -99.0322495, 90.8628616, -189.3471527, 189.5896301
3: -114.5677872, 117.5146637, -115.2665710, 118.0243607, -232.5921326, 232.7812347
4: -121.6564331, 111.7990570, -122.3268661, 112.1714935, -233.8279266, 234.1259155
5: -110.6400604, 117.8679428, -111.1869507, 118.3193054, -228.9593506, 229.0548859
6: -137.5158997, 120.3617401, -137.7515869, 120.7258911, -258.2417603, 258.1133423
7: -134.8545532, 119.3860931, -135.3386078, 119.6289291, -254.4834900, 254.7247009
8: -126.7255859, 132.4398193, -127.3208618, 132.8108215, -259.5364075, 259.7606812
9: -118.8068085, 109.9080811, -119.1847153, 110.2141953, -229.0209961, 229.0928040
10: -173.7842712, 162.0623169, -174.4564667, 163.2332611, -337.0174561, 336.5187988
11: -166.1309052, 128.3669128, -166.8192749, 129.4568787, -295.5877686, 295.1861877
12: -155.3451996, 145.5450287, -156.0345154, 146.5398560, -301.8850708, 301.5795288
13: -160.7512817, 167.9092712, -161.4120483, 168.4318542, -329.1831360, 329.3213196
14: -238.0004883, 145.9578552, -238.8528442, 146.8846130, -384.8851013, 384.8106995
15: -138.4383240, 111.6974487, -139.1199799, 111.9814911, -250.4198151, 250.8174133
16: -181.7315674, 133.5110779, -182.2665405, 134.2841797, -316.0156860, 315.7776184
17: -248.7440643, 194.5653534, -249.5715790, 195.8249817, -444.5690308, 444.1369324
18: -152.0092010, 125.4978790, -152.5308838, 126.3327713, -278.3419800, 278.0287476
19: -116.7126541, 74.3825836, -117.1643524, 74.8650818, -191.5777283, 191.5469360
20: -109.6524963, 92.4611664, -110.0064621, 92.8816376, -202.5341339, 202.4675903
21: -148.4987640, 99.6449738, -149.0252075, 100.3439560, -248.8427124, 248.6701813
22: -157.7925415, 107.4088287, -158.1621857, 107.9179840, -265.7105103, 265.5710144
23: -119.7585754, 94.4587250, -120.1907043, 94.9988556, -214.7574310, 214.6494293
24: -147.2999573, 108.3781128, -147.7488098, 108.7932663, -256.0932312, 256.1269226
25: -123.8804703, 105.6260529, -124.2766724, 106.1147232, -229.9951782, 229.9027252
26: -170.4028320, 147.5608826, -171.0301056, 148.5114136, -318.9142456, 318.5910034
27: -154.9410706, 110.6074066, -155.3030090, 110.9129791, -265.8540649, 265.9104004
28: -115.8267670, 99.7016830, -116.1272888, 100.0054016, -215.8321686, 215.8289490
29: -173.9377441, 115.0544586, -174.3394623, 115.7367554, -289.6744995, 289.3939209
30: -146.3535004, 124.7689056, -146.7805786, 125.5476227, -271.9011230, 271.5494995
31: -149.3532104, 101.1433334, -149.9263000, 101.7531586, -251.1063690, 251.0696106
32: -146.2322388, 116.7891159, -146.5553284, 117.0778122, -263.3100586, 263.3444519
33: -191.8767242, 158.7216187, -192.6051788, 159.1812134, -351.0579224, 351.3267822
34: -160.9298859, 123.9075241, -161.4227600, 124.2248154, -285.1546936, 285.3302917
35: -162.4543152, 126.3980789, -163.1442719, 126.7807236, -289.2349854, 289.5423584
36: -153.6648712, 124.8371429, -154.2060852, 125.1017303, -278.7666016, 279.0432129
37: -223.1360931, 141.5203857, -223.6739502, 141.8978882, -365.0339355, 365.1943359
38: -189.6068726, 154.5263062, -190.2820282, 154.8475037, -344.4543762, 344.8082886
39: -225.5913849, 153.8287659, -226.3135376, 154.1844788, -379.7758789, 380.1423035
40: -191.6238251, 133.3120422, -192.1572571, 133.5318604, -325.1556702, 325.4692993
41: -143.9954071, 110.9291992, -144.3283386, 111.2322998, -255.2276764, 255.2575378
42: -110.8106079, 108.2577133, -111.0910339, 108.8837814, -219.6943817, 219.3487549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=698, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1162
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1225
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1175
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 264
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 308
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 325
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 291
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1178
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 257
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 321
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 336
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 305
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 1792
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 640

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0879944, upper bound: 150.0879945
time: 376.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
time: 156.03 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -194.3078918, 123.7332611, -194.3560638, 123.7643051, -318.0722046, 318.0893250
1: -112.6309509, 98.2404327, -112.6573334, 98.2656555, -210.8966064, 210.8977661
2: -99.0969391, 90.8699188, -99.1251068, 90.8807373, -189.9776764, 189.9950256
3: -115.3410339, 118.0351181, -115.3852997, 118.0533752, -233.3944092, 233.4204102
4: -122.3941040, 112.1770630, -122.4373779, 112.1905746, -234.5846863, 234.6144104
5: -111.2417908, 118.3307190, -111.2773132, 118.3461990, -229.5879822, 229.6080322
6: -137.7627258, 120.6949234, -137.7839966, 120.7834320, -258.5461426, 258.4789124
7: -135.3790588, 119.6356812, -135.4126282, 119.6500015, -255.0290527, 255.0483093
8: -127.3813705, 132.8206482, -127.4194870, 132.8383179, -260.2196960, 260.2400818
9: -119.1719818, 110.2248077, -119.2357254, 110.2499542, -229.4219208, 229.4605408
10: -174.4777222, 163.3535614, -174.5004272, 163.4277802, -337.9054871, 337.8539734
11: -166.8300781, 129.5824585, -166.8539734, 129.6512146, -296.4812622, 296.4364319
12: -156.0473022, 146.6483459, -156.0662231, 146.7076263, -302.7549438, 302.7145691
13: -161.4288025, 168.4491882, -161.5158081, 168.4806824, -329.9094849, 329.9649963
14: -238.8775940, 146.9903412, -238.9123383, 147.0448761, -385.9224243, 385.9026794
15: -139.2107544, 111.9941559, -139.2399445, 112.0155334, -251.2262878, 251.2341003
16: -182.2944336, 134.3719482, -182.3288879, 134.4130859, -316.7075195, 316.7008362
17: -249.5905762, 195.9656677, -249.6140747, 196.0419617, -445.6325378, 445.5797424
18: -152.5502014, 126.4285126, -152.5888062, 126.4845505, -279.0347595, 279.0173340
19: -117.1812973, 74.9178162, -117.1988754, 74.9498978, -192.1311951, 192.1166992
20: -110.0221634, 92.9320831, -110.0395050, 92.9520416, -202.9742126, 202.9715881
21: -149.0392609, 100.4221573, -149.0595856, 100.4649124, -249.5041351, 249.4817047
22: -158.1729431, 107.9643173, -158.1970825, 108.0006943, -266.1736145, 266.1613770
23: -120.2049866, 95.0537720, -120.2197723, 95.0907288, -215.2956848, 215.2735291
24: -147.7589722, 108.7959595, -147.7882538, 108.8617859, -256.6207581, 256.5842285
25: -124.2901840, 106.1631775, -124.3082123, 106.1959991, -230.4861755, 230.4713898
26: -171.0451050, 148.6124725, -171.0725708, 148.6687469, -319.7138672, 319.6850586
27: -155.3201752, 110.8967896, -155.3504333, 110.9619522, -266.2821350, 266.2472229
28: -116.1407166, 100.0255280, -116.1600723, 100.0528336, -216.1935272, 216.1856079
29: -174.3449860, 115.8067093, -174.3669586, 115.8534164, -290.1983948, 290.1736755
30: -146.7911530, 125.6270905, -146.8119354, 125.6781540, -272.4692993, 272.4389954
31: -149.9496460, 101.8212433, -149.9770966, 101.8592148, -251.8088684, 251.7983246
32: -146.5530548, 117.0799866, -146.5981750, 117.1200714, -263.6731262, 263.6781616
33: -192.6744690, 159.1912537, -192.7234802, 159.2073059, -351.8817749, 351.9147339
34: -161.4667969, 124.2390137, -161.5019989, 124.2549896, -285.7217712, 285.7410278
35: -163.2134094, 126.7888718, -163.2583618, 126.8004913, -290.0139160, 290.0472412
36: -154.2553101, 125.1101837, -154.2919922, 125.1236725, -279.3789368, 279.4021606
37: -223.6952362, 141.8906860, -223.7451019, 141.9507141, -365.6459351, 365.6357727
38: -190.3352661, 154.8501587, -190.3866577, 154.8811646, -345.2163696, 345.2368164
39: -226.3760681, 154.1922302, -226.4278870, 154.2048950, -380.5809631, 380.6201172
40: -192.1906281, 133.5386047, -192.2375641, 133.5492554, -325.7398682, 325.7761536
41: -144.3476410, 111.2127991, -144.3735046, 111.2714844, -255.6190948, 255.5862885
42: -111.1004257, 108.9377899, -111.1141891, 108.9842834, -220.0847015, 220.0519714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=698, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1239
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1238
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1241
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1245
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1240
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1235
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1171
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1234
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1237
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1155
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1138
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 1256
type: B, layer: 1, pos: 1108
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1080
type: B, layer: 1, pos: 1218
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1181
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1219
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1154
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1166
type: B, layer: 1, pos: 1251
type: B, layer: 1, pos: 1236
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1079
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1222
type: B, layer: 1, pos: 1272
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1182
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1064
type: B, layer: 1, pos: 1170
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 1093
type: B, layer: 1, pos: 1187
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1214
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1165
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1100
type: B, layer: 1, pos: 1257
type: B, layer: 1, pos: 1109
type: B, layer: 1, pos: 1223
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1099
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1116
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1244
type: B, layer: 1, pos: 1221
type: B, layer: 1, pos: 1254
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1267
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1081
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1046
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1159
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1115
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1213
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1077
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1092
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1220
type: B, layer: 1, pos: 1252
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1149
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1203
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1150
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1258
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1076
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1183
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1065
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1278
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1133
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1073
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1091
type: B, layer: 1, pos: 1226
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1162
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1225
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1268
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1075
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1227
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1205
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1059
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1089
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1105
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1045
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1094
type: B, layer: 1, pos: 1083
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1053
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 1104
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1228
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1175
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1060
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1204
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 1084
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1058
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1140
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1269
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1253
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1082
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1088
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1211
type: B, layer: 1, pos: 1062
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1044
type: B, layer: 1, pos: 1098
type: B, layer: 1, pos: 263
type: B, layer: 1, pos: 1057
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1056
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1061
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1041
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 260
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1074
type: B, layer: 1, pos: 1042
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1072
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1025
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 262
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 324
type: B, layer: 1, pos: 323
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1224
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 264
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 308
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 1066
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 325
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1110
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 291
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1043
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1097
type: B, layer: 1, pos: 1078
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 256
type: B, layer: 1, pos: 1197
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1178
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1095
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 322
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 257
type: B, layer: 1, pos: 1040
type: B, layer: 1, pos: 321
type: B, layer: 1, pos: 277
type: B, layer: 1, pos: 336
type: B, layer: 1, pos: 294
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 258
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 292
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1212
type: B, layer: 1, pos: 305
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 337
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 1196
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 353
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 293
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 306
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 1792
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 640

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.1290824, upper bound: 150.0879946
time: 346.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.1290824, upper bound: 150.1290825
time: 515.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 864.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 864.23
Output dim: 3, lower bound: -150.0879944, upper bound: 150.0879945
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 864.23
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 864.23
Output dim: 3, lower bound: -150.1290824, upper bound: 150.0879946
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 864.23
Output dim: 3, lower bound: -150.1290824, upper bound: 150.1290825

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -193.7549591, 123.4849701, -193.7549591, 123.4849701, -317.2399292, 317.2399292
1: -112.2062836, 98.0295868, -112.2062836, 98.0295868, -210.2358704, 210.2358704
2: -98.4843063, 90.5573730, -98.4843063, 90.5573730, -189.0416718, 189.0416870
3: -114.5677872, 117.5146637, -114.5677872, 117.5146637, -232.0824280, 232.0824432
4: -121.6564331, 111.7990570, -121.6564331, 111.7990570, -233.4554749, 233.4554749
5: -110.6400604, 117.8679428, -110.6400604, 117.8679428, -228.5079956, 228.5079956
6: -137.5158997, 120.3617401, -137.5158997, 120.3617401, -257.8776245, 257.8776245
7: -134.8545532, 119.3860931, -134.8545532, 119.3860931, -254.2406464, 254.2406464
8: -126.7255859, 132.4398193, -126.7255859, 132.4398193, -259.1654053, 259.1654053
9: -118.8068085, 109.9080811, -118.8068085, 109.9080811, -228.7148895, 228.7148895
10: -173.7842712, 162.0623169, -173.7842712, 162.0623169, -335.8465881, 335.8465881
11: -166.1309052, 128.3669128, -166.1309052, 128.3669128, -294.4978027, 294.4978027
12: -155.3451996, 145.5450287, -155.3451996, 145.5450287, -300.8902283, 300.8901978
13: -160.7512817, 167.9092712, -160.7512817, 167.9092712, -328.6605530, 328.6605225
14: -238.0004883, 145.9578552, -238.0004883, 145.9578552, -383.9583435, 383.9583435
15: -138.4383240, 111.6974487, -138.4383240, 111.6974487, -250.1357727, 250.1357727
16: -181.7315674, 133.5110779, -181.7315674, 133.5110779, -315.2426453, 315.2426147
17: -248.7440643, 194.5653534, -248.7440643, 194.5653534, -443.3094177, 443.3094177
18: -152.0092010, 125.4978790, -152.0092010, 125.4978790, -277.5070801, 277.5070801
19: -116.7126541, 74.3825836, -116.7126541, 74.3825836, -191.0952148, 191.0952148
20: -109.6524963, 92.4611664, -109.6524963, 92.4611664, -202.1136627, 202.1136475
21: -148.4987640, 99.6449738, -148.4987640, 99.6449738, -248.1437378, 248.1437378
22: -157.7925415, 107.4088287, -157.7925415, 107.4088287, -265.2013550, 265.2013550
23: -119.7585754, 94.4587250, -119.7585754, 94.4587250, -214.2173004, 214.2173004
24: -147.2999573, 108.3781128, -147.2999573, 108.3781128, -255.6780701, 255.6780701
25: -123.8804703, 105.6260529, -123.8804703, 105.6260529, -229.5065155, 229.5065308
26: -170.4028320, 147.5608826, -170.4028320, 147.5608826, -317.9637146, 317.9637146
27: -154.9410706, 110.6074066, -154.9410706, 110.6074066, -265.5484619, 265.5484619
28: -115.8267670, 99.7016830, -115.8267670, 99.7016830, -215.5284424, 215.5284424
29: -173.9377441, 115.0544586, -173.9377441, 115.0544586, -288.9921875, 288.9921875
30: -146.3535004, 124.7689056, -146.3535004, 124.7689056, -271.1224060, 271.1224060
31: -149.3532104, 101.1433334, -149.3532104, 101.1433334, -250.4965515, 250.4965363
32: -146.2322388, 116.7891159, -146.2322388, 116.7891159, -263.0213318, 263.0213318
33: -191.8767242, 158.7216187, -191.8767242, 158.7216187, -350.5983276, 350.5982971
34: -160.9298859, 123.9075241, -160.9298859, 123.9075241, -284.8374023, 284.8374023
35: -162.4543152, 126.3980789, -162.4543152, 126.3980789, -288.8523560, 288.8523560
36: -153.6648712, 124.8371429, -153.6648712, 124.8371429, -278.5020142, 278.5020142
37: -223.1360931, 141.5203857, -223.1360931, 141.5203857, -364.6564636, 364.6564941
38: -189.6068726, 154.5263062, -189.6068726, 154.5263062, -344.1331787, 344.1331787
39: -225.5913849, 153.8287659, -225.5913849, 153.8287659, -379.4201355, 379.4201355
40: -191.6238251, 133.3120422, -191.6238251, 133.3120422, -324.9358521, 324.9358521
41: -143.9954071, 110.9291992, -143.9954071, 110.9291992, -254.9245911, 254.9246063
42: -110.8106079, 108.2577133, -110.8106079, 108.2577133, -219.0682983, 219.0682983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=697, inp2_unstable=697, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1162
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1225
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1175
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1178
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 257
type: A, layer: 1, pos: 321
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 336
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 1792
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 640

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0394474, upper bound: 150.0870454
time: 932.97 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0394474, upper bound: 150.0870454
time: 407.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 1342.98 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1342.98
Output dim: 3, lower bound: -150.0394474, upper bound: 150.0870454
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1342.98
Output dim: 3, lower bound: -150.0394474, upper bound: 150.0870454
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1342.98
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1342.98
Output dim: 3, lower bound: -150.1290824, upper bound: 150.0879946
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1342.98
Output dim: 3, lower bound: -150.1290824, upper bound: 150.1290825
Binary search (step 0): status=Status.UNKNOWN, k_low=8, k_high=12, k_mid=10, eps_mid=0.0390625, abs_max=233.43865966796875
rel_dist={3: [-150.13758384939092, 150.13758388525406]}

## Binary search (step 1) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1239
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1238
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1241
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1245
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1240
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1235
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1171
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1234
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1237
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1155
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1138
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1256
type: A, layer: 1, pos: 1108
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1080
type: A, layer: 1, pos: 1218
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1181
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1219
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1154
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1166
type: A, layer: 1, pos: 1251
type: A, layer: 1, pos: 1236
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1079
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1222
type: A, layer: 1, pos: 1272
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1182
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1064
type: A, layer: 1, pos: 1170
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 1093
type: A, layer: 1, pos: 1187
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1214
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1165
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1100
type: A, layer: 1, pos: 1257
type: A, layer: 1, pos: 1109
type: A, layer: 1, pos: 1223
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1099
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 1116
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1244
type: A, layer: 1, pos: 1221
type: A, layer: 1, pos: 1254
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1267
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1081
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1046
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1159
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1115
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1213
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1077
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1092
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1220
type: A, layer: 1, pos: 1252
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1149
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1203
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1150
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1258
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1076
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1183
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1065
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1278
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1133
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1073
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1091
type: A, layer: 1, pos: 1226
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1162
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1225
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1268
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1075
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1227
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1205
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1059
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1089
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1105
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1045
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1094
type: A, layer: 1, pos: 1083
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1053
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 1104
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1228
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1175
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1060
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1204
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 1084
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1058
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1140
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1269
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1253
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1082
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1088
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1211
type: A, layer: 1, pos: 1062
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1044
type: A, layer: 1, pos: 1098
type: A, layer: 1, pos: 263
type: A, layer: 1, pos: 1057
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1056
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1061
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1041
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 260
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1074
type: A, layer: 1, pos: 1042
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1072
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1025
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 262
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 324
type: A, layer: 1, pos: 323
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1224
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 264
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 1066
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 325
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1110
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 291
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1043
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1097
type: A, layer: 1, pos: 1078
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 256
type: A, layer: 1, pos: 1197
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1178
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 1095
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 322
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 257
type: A, layer: 1, pos: 1040
type: A, layer: 1, pos: 321
type: A, layer: 1, pos: 277
type: A, layer: 1, pos: 336
type: A, layer: 1, pos: 294
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 258
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 292
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1212
type: A, layer: 1, pos: 305
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 337
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 1196
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 353
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 293
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 306
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 1792
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 640

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.5611855, upper bound: 148.6007015
time: 1071.40 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.5611855, upper bound: 148.6007015
time: 7022.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8094.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8094.16
Output dim: 3, lower bound: -148.5611855, upper bound: 148.6007015
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8094.16
Output dim: 3, lower bound: -148.5611855, upper bound: 148.6007015
Binary search (step 1): status=Status.UNKNOWN, k_low=8, k_high=9, k_mid=8, eps_mid=0.0312500, abs_max=233.43865966796875
rel_dist={3: [-148.6090126930012, 148.60901277819403]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.02734375
execution time: 13443.89 seconds
